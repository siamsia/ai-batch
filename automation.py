#!/usr/bin/env python3
import os, time, json, uuid, shutil, subprocess, random, re
import requests
from pathlib import Path
from urllib.parse import quote
from datetime import datetime

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ====== Paths / Consts ======
BASE_DIR   = Path("/workspace")
COMFY_DIR  = BASE_DIR/"ComfyUI"
OUT_ROOT   = BASE_DIR/"outputs"
LOG_DIR    = BASE_DIR/"logs"
MODEL_DIR  = BASE_DIR/"models"/"checkpoints"
CFG_DIR    = BASE_DIR/"config"
CTRL_DIR   = BASE_DIR/".agent"

CONFIG_FILE = CFG_DIR/"run_config.jsonc"   # << JSONC!

API_BASE   = os.getenv("PROMPT_API_BASE", "https://manus-api-server.onrender.com")
GET_NEXT   = f"{API_BASE}/get_next_prompt"
MARK_USED  = f"{API_BASE}/mark_prompt_used"
MARK_LOCK  = f"{API_BASE}/mark_prompt_locked"

HOST="127.0.0.1"; PORT=int(os.getenv("COMFY_PORT","8188")); BASE=f"http://{HOST}:{PORT}"
CANCEL_FLAG = CTRL_DIR/"CANCEL"

TOKEN_PATH  = CFG_DIR/"token.json"
SCOPES = ['https://www.googleapis.com/auth/drive.file']
GDRIVE_ROOT_NAME = "output"

# ====== Util: JSONC loader ======
def load_jsonc(path: Path):
    txt = path.read_text(encoding="utf-8")
    # ตัด // ... และ /* ... */ ออกให้กลายเป็น JSON ปกติ
    txt = re.sub(r"//.*?$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.DOTALL)
    return json.loads(txt)

# ====== HTTP helpers ======
def get_json(url, tries=5, backoff=2, timeout=30):
    last=None
    for i in range(tries):
        try:
            r=requests.get(url,timeout=timeout)
            if r.status_code==200: return r.json()
            last=f"{r.status_code} {r.text[:200]}"
        except Exception as e: last=str(e)
        time.sleep(backoff*(i+1))
    raise RuntimeError(f"GET {url} failed: {last}")

def post_json(url, payload, tries=5, backoff=2, timeout=60):
    last=None
    for i in range(tries):
        try:
            r=requests.post(url,json=payload,timeout=timeout)
            if r.status_code==200: return r.json()
            last=f"{r.status_code} {r.text[:200]}"
        except Exception as e: last=str(e)
        time.sleep(backoff*(i+1))
    raise RuntimeError(f"POST {url} failed: {last}")

# ====== ComfyUI helpers ======
def comfy_ready(timeout=180):
    t0=time.time()
    while time.time()-t0<timeout:
        try:
            if requests.get(f"{BASE}/object_info",timeout=5).status_code==200:
                return True
        except: pass
        time.sleep(2)
    raise TimeoutError("ComfyUI not ready")

def submit(graph):
    cid=str(uuid.uuid4())
    r=requests.post(f"{BASE}/prompt",json={"prompt":graph,"client_id":cid},timeout=60)
    if r.status_code!=200:
        print("ERR /prompt:",r.status_code,r.text[:200]); r.raise_for_status()
    return r.json()["prompt_id"]

def wait_outputs(pid, timeout=1800):
    t0=time.time()
    while time.time()-t0<timeout:
        try:
            h=requests.get(f"{BASE}/history/{pid}",timeout=15).json().get(pid,{})
            outs=h.get("outputs") or {}
            if any(x.get("images") for x in outs.values()): return h
            if h.get("node_errors"): print("node_errors:",h["node_errors"]); return h
        except: time.sleep(2)
        time.sleep(2)
    raise TimeoutError("history timeout")

def fetch_images(pid, save_to:Path):
    save_to.mkdir(parents=True,exist_ok=True); saved=[]
    h=requests.get(f"{BASE}/history/{pid}",timeout=30).json().get(pid,{})
    for node_out in (h.get("outputs") or {}).values():
        for img in node_out.get("images",[]):
            filename=img["filename"]; sub=img.get("subfolder","") or ""
            typ=img.get("type","output") or "output"
            for sf in [sub, f"user/default/{sub}" if sub else "", "user/default", ""]:
                url=f"{BASE}/view?filename={quote(filename)}&subfolder={quote(sf)}&type={quote(typ)}"
                try:
                    r=requests.get(url,timeout=60)
                    if r.status_code==200:
                        outp=save_to/filename; open(outp,"wb").write(r.content); saved.append(str(outp)); break
                except: time.sleep(1)
    return saved

# ====== Real-ESRGAN ======
def upscale(src, dst, model_name, scale):
    cmd=[str(BASE_DIR/"venv"/"bin"/"python"), "-m","realesrgan",
         "-n", model_name, "-s", str(scale), "-i", src, "-o", dst]
    subprocess.check_call(cmd, timeout=900)

# ====== Metadata ======
def embed_meta(img,title,keywords):
    if isinstance(keywords,(list,tuple)): kw=", ".join([k for k in keywords if k])
    else: kw=str(keywords)
    try:
        subprocess.call(["exiftool","-overwrite_original",
                         f"-XMP-dc:Title={title}",
                         f"-IPTC:Keywords={kw}",
                         f"-XMP-dc:Subject={kw}", img],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print("[WARN] exif:",e)

# ====== Workflow builder ======
def build_wf(ckpt_name, prompt, neg, w, h, steps, cfg, sampler, scheduler):
    seed = random.randint(0, 2**31-1)
    return {
      "1":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":ckpt_name}},
      "2":{"class_type":"CLIPTextEncode","inputs":{"clip":["1",1],"text":prompt}},
      "3":{"class_type":"CLIPTextEncode","inputs":{"clip":["1",1],"text":neg}},
      "4":{"class_type":"EmptyLatentImage","inputs":{"batch_size":1,"width":w,"height":h}},
      "5":{"class_type":"KSampler","inputs":{
        "model":["1",0],"positive":["2",0],"negative":["3",0],"latent_image":["4",0],
        "seed":seed,"steps":steps,"cfg":cfg,"sampler_name":sampler,"scheduler":scheduler,"denoise":1.0}},
      "6":{"class_type":"VAEDecode","inputs":{"samples":["5",0],"vae":["1",2]}},
      "7":{"class_type":"SaveImage","inputs":{"images":["6",0],"filename_prefix":"gen"}}
    }

# ====== Model auto-downloader ======
def ensure_model(cfg) -> str:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ตัดสินใจจาก config
    model = (cfg.get("model") or "SDXL").upper()
    source = (cfg.get("model_source") or "AUTO").upper()
    filename = cfg.get("model_filename") or ("sd_xl_base_1.0.safetensors" if model=="SDXL" else "v1-5-pruned-emaonly.safetensors")
    target = MODEL_DIR/filename
    if target.exists(): 
        return filename

    def hf_url(repo, path):
        return f"https://huggingface.co/{repo}/resolve/main/{path}"

    url = None
    headers = {}

    if source == "AUTO":
        if model == "SDXL":
            url = hf_url("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors")
            filename = "sd_xl_base_1.0.safetensors"
            target = MODEL_DIR/filename
        elif model == "SD15":
            url = hf_url("runwayml/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors")
            filename = "v1-5-pruned-emaonly.safetensors"
            target = MODEL_DIR/filename
        else:
            raise RuntimeError("AUTO ไม่รู้จักโมเดลนี้ (ใช้ HUGGINGFACE หรือ DIRECT แทน)")
    elif source == "HUGGINGFACE":
        repo = cfg.get("hf_repo"); path = cfg.get("hf_path")
        if not repo or not path:
            raise RuntimeError("ต้องระบุ hf_repo และ hf_path เมื่อใช้ model_source=HUGGINGFACE")
        url = hf_url(repo, path)
        tok_env = cfg.get("hf_token_env") or ""
        if tok_env and os.getenv(tok_env):
            headers["Authorization"] = f"Bearer {os.getenv(tok_env)}"
    elif source == "DIRECT":
        url = cfg.get("direct_url")
        if not url:
            raise RuntimeError("ต้องระบุ direct_url เมื่อใช้ model_source=DIRECT")
    else:
        raise RuntimeError("model_source ไม่ถูกต้อง")

    # ดาวน์โหลด
    print(f"[*] Download model -> {target.name}")
    cmd = ["wget","-O",str(target), url]
    if headers:
        for k,v in headers.items():
            cmd[1:1] = ["--header", f"{k}: {v}"]
    subprocess.check_call(cmd, timeout=3600)

    if not target.exists() or target.stat().st_size < 10*1024*1024:
        raise RuntimeError("ดาวน์โหลดโมเดลไม่สมบูรณ์")
    return target.name

# ====== Google Drive ======
def drive():
    if not TOKEN_PATH.exists(): raise RuntimeError(f"token.json not found at {TOKEN_PATH}")
    creds=Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    return build('drive','v3',credentials=creds,cache_discovery=False)

def gdrive_find_folder(svc,name,parent=None):
    q=f"name='{name.replace(\"'\",\"\\'\")}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent: q+=f" and '{parent}' in parents"
    res=svc.files().list(q=q,spaces='drive',fields="files(id,name)",pageSize=10).execute()
    fs=res.get('files',[]); return fs[0]['id'] if fs else None

def gdrive_make_folder(svc,name,parent=None):
    meta={'name':name,'mimeType':'application/vnd.google-apps.folder'}
    if parent: meta['parents']=[parent]
    return svc.files().create(body=meta,fields='id').execute()['id']

def gdrive_ensure_folder(svc,name,parent=None):
    return gdrive_find_folder(svc,name,parent) or gdrive_make_folder(svc,name,parent)

def gdrive_upload_folder(svc, local_dir:Path, parent_id:str):
    uploaded=[]; date_id=gdrive_ensure_folder(svc, local_dir.name, parent_id)
    for p in sorted(local_dir.glob("*")):
        if p.is_file():
            media=MediaFileUpload(str(p),resumable=True)
            meta={'name':p.name,'parents':[date_id]}
            f=svc.files().create(body=meta,media_body=media,fields='id').execute()
            uploaded.append((p,f['id']))
    return uploaded

# ====== Main ======
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CTRL_DIR.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)

    # โหลด config (JSONC)
    cfg = load_jsonc(CONFIG_FILE)

    # ตรวจ/โหลดโมเดลตาม config
    ckpt_name = ensure_model(cfg)  # คืนชื่อไฟล์ใน MODEL_DIR

    # โฟลเดอร์ผลลัพธ์แยกตามวันที่
    date = datetime.now().strftime("%Y-%m-%d")
    OUT_DIR = OUT_ROOT/date
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ดึงงาน
    batch = get_json(GET_NEXT)
    items = (batch or {}).get("prompts", [])
    if not items:
        print("[INFO] no prompts")
        return

    log_id = str(uuid.uuid4())
    try:
        post_json(MARK_LOCK, {"rowId": items[0]["rowId"], "log_id": log_id})
    except Exception as e:
        print("[WARN] lock:", e)

    # รอ Comfy พร้อม
    comfy_ready()

    # อ่านพารามิเตอร์เจนจาก config
    W  = int(cfg.get("gen_width", 1024))
    H  = int(cfg.get("gen_height", 1024))
    ST = int(cfg.get("gen_steps", 28))
    CF = float(cfg.get("gen_cfg", 7.0))
    SP = cfg.get("sampler", "dpmpp_2m")
    SC = cfg.get("scheduler", "karras")
    NEG= cfg.get("neg_prompt", "text, watermark, logo, signature, blurry, noisy")

    UPSCALE = bool(cfg.get("upscale", True))
    UPSCALE_MODEL = cfg.get("upscale_model", "RealESRGAN_x2plus")
    UPSCALE_SCALE = int(cfg.get("upscale_scale", 2))
    OUT_SUFFIX = cfg.get("output_suffix", "_2k")
    TARGET_PIX = int(cfg.get("target_pixels", 0))

    done = []
    for i, it in enumerate(items, 1):
        if CANCEL_FLAG.exists():
            print("[CANCEL] stop."); break

        topic  = (it.get("topic") or "").strip()
        prompt = (it.get("prompt") or "").strip()
        title  = (it.get("title")  or topic or prompt)[:70].strip()
        kws    = it.get("keywords") or []
        if not kws:
            seen=set(); kws=[]
            for tok in prompt.split():
                t=tok.strip(",.;:()[]{}'\"").lower()
                if t and t not in seen:
                    seen.add(t); kws.append(t)
                if len(kws)>=10: break

        print(f"=== [{i}/{len(items)}] rowId={it.get('rowId')} {topic} ===")
        pid = submit(build_wf(ckpt_name, prompt, NEG, W, H, ST, CF, SP, SC))
        _   = wait_outputs(pid)

        tmp = OUT_DIR/"tmp"; tmp.mkdir(parents=True, exist_ok=True)
        imgs = fetch_images(pid, tmp)
        if not imgs:
            print("[WARN] no image"); shutil.rmtree(tmp, ignore_errors=True); continue

        src = imgs[0]; base = Path(src).stem

        # อัปสเกลตาม config
        if UPSCALE:
            dst = OUT_DIR / f"{base}{OUT_SUFFIX}.jpg"
            upscale(src, str(dst), UPSCALE_MODEL, UPSCALE_SCALE)
        else:
            dst = OUT_DIR / f"{base}.jpg"
            shutil.copy(src, dst)

        embed_meta(str(dst), title, kws)
        print("  - saved:", dst)
        done.append(it.get("rowId"))
        shutil.rmtree(tmp, ignore_errors=True)

    if done:
        print("[*] mark used…", post_json(MARK_USED, {"rowIds": done, "log_id": log_id}))

    # อัปโหลดโฟลเดอร์วันที่ขึ้น Google Drive/output
    try:
        svc = drive()
        root_id = gdrive_ensure_folder(svc, GDRIVE_ROOT_NAME)
        up = gdrive_upload_folder(svc, OUT_DIR, root_id)
        print(f"[*] uploaded {len(up)} files -> Drive/{GDRIVE_ROOT_NAME}/{OUT_DIR.name}")
        if up and len(list(OUT_DIR.glob("*"))) >= len(up):
            shutil.rmtree(OUT_DIR); print("[*] removed local date folder")
    except Exception as e:
        print("[WARN] drive upload failed:", e)

if __name__ == "__main__":
    main()
