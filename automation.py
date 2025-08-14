#!/usr/bin/env python3
import os, time, json, uuid, shutil, subprocess, random
import requests
from pathlib import Path
from urllib.parse import quote
from datetime import datetime

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ======== CONFIG ========
API_BASE = os.getenv("PROMPT_API_BASE", "https://manus-api-server.onrender.com")
GET_NEXT  = f"{API_BASE}/get_next_prompt"
MARK_USED = f"{API_BASE}/mark_prompt_used"
MARK_LOCK = f"{API_BASE}/mark_prompt_locked"

HOST="127.0.0.1"; PORT=int(os.getenv("COMFY_PORT","8188")); BASE=f"http://{HOST}:{PORT}"
COMFY_DIR = Path("/workspace/ComfyUI")
OUT_ROOT  = Path("/workspace/outputs")
LOG_DIR   = Path("/workspace/logs")
MODEL_DIR = Path("/workspace/models/checkpoints")

WIDTH=int(os.getenv("GEN_WIDTH","2048"))
HEIGHT=int(os.getenv("GEN_HEIGHT","2048"))
STEPS=int(os.getenv("GEN_STEPS","24"))
CFG=float(os.getenv("GEN_CFG","7.0"))
SAMPLER=os.getenv("GEN_SAMPLER","dpmpp_2m")
SCHED=os.getenv("GEN_SCHEDULER","karras")

CTRL_DIR=Path("/workspace/.agent"); CANCEL_FLAG=CTRL_DIR/"CANCEL"

TOKEN_PATH=Path("/workspace/config/token.json")
GDRIVE_ROOT_NAME="output"
SCOPES=['https://www.googleapis.com/auth/drive.file']

# ======== HTTP helpers ========
def get_json(url, t=5, backoff=2, timeout=30):
    last=None
    for i in range(t):
        try:
            r=requests.get(url,timeout=timeout); 
            if r.status_code==200: return r.json()
            last=f"{r.status_code} {r.text[:200]}"
        except Exception as e: last=str(e)
        time.sleep(backoff*(i+1))
    raise RuntimeError(f"GET {url} failed: {last}")

def post_json(url, payload, t=5, backoff=2, timeout=60):
    last=None
    for i in range(t):
        try:
            r=requests.post(url,json=payload,timeout=timeout)
            if r.status_code==200: return r.json()
            last=f"{r.status_code} {r.text[:200]}"
        except Exception as e: last=str(e)
        time.sleep(backoff*(i+1))
    raise RuntimeError(f"POST {url} failed: {last}")

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

def upscale_4k(src, dst, scale="2", model="RealESRGAN_x4plus"):
    cmd=[str(Path("/workspace/venv/bin/python")), "-m","realesrgan","-n",model,"-s",scale,"-i",src,"-o",dst]
    subprocess.check_call(cmd,timeout=900)

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

def pick_ckpt():
    files=sorted(MODEL_DIR.glob("*.safetensors"))
    if not files: raise RuntimeError("No model in /workspace/models/checkpoints")
    return files[0].name

def build_wf(ckpt,prompt,neg,w,h):
    import time as _t, random as _r
    seed=(int(_t.time()) ^ _r.randint(0,2**31-1)) & 0x7FFFFFFF
    return {
      "1":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":ckpt}},
      "2":{"class_type":"CLIPTextEncode","inputs":{"clip":["1",1],"text":prompt}},
      "3":{"class_type":"CLIPTextEncode","inputs":{"clip":["1",1],"text":neg}},
      "4":{"class_type":"EmptyLatentImage","inputs":{"batch_size":1,"width":w,"height":h}},
      "5":{"class_type":"KSampler","inputs":{
        "model":["1",0],"positive":["2",0],"negative":["3",0],"latent_image":["4",0],
        "seed":seed,"steps":STEPS,"cfg":CFG,"sampler_name":SAMPLER,"scheduler":SCHED,"denoise":1.0}},
      "6":{"class_type":"VAEDecode","inputs":{"samples":["5",0],"vae":["1",2]}},
      "7":{"class_type":"SaveImage","inputs":{"images":["6",0],"filename_prefix":"gen2048"}}
    }

# ---- Google Drive ----
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

# -------------- main --------------
def main():
    OUT_ROOT.mkdir(parents=True,exist_ok=True); LOG_DIR.mkdir(parents=True,exist_ok=True)
    date=datetime.now().strftime("%Y-%m-%d"); OUT_DIR=OUT_ROOT/date; OUT_DIR.mkdir(parents=True,exist_ok=True)

    batch=get_json(GET_NEXT); items=(batch or {}).get("prompts",[])
    if not items: print("[INFO] no prompts"); return

    log_id=str(uuid.uuid4()); print(f"[*] {len(items)} prompts, log_id={log_id}")
    try: post_json(MARK_LOCK, {"rowId":items[0]["rowId"],"log_id":log_id})
    except Exception as e: print("[WARN] lock:",e)

    # ensure comfy
    comfy_ready()

    ckpt=pick_ckpt(); neg="text, watermark, logo, signature, blurry, noisy"
    done=[]

    for i,it in enumerate(items,1):
        if CANCEL_FLAG.exists():
            print("[CANCEL] stop."); break
        topic=(it.get("topic") or "").strip()
        prompt=(it.get("prompt") or "").strip()
        title=(it.get("title") or topic or prompt)[:70].strip()
        kws=it.get("keywords") or []
        if not kws:
            seen=set(); kws=[]
            for tok in prompt.split():
                t=tok.strip(",.;:()[]{}'\"").lower()
                if t and t not in seen:
                    seen.add(t); kws.append(t)
                if len(kws)>=10: break

        print(f"=== [{i}/{len(items)}] rowId={it.get('rowId')} {topic} ===")
        pid=submit(build_wf(ckpt,prompt,neg,WIDTH,HEIGHT))
        _=wait_outputs(pid)

        tmp=OUT_DIR/"tmp"; tmp.mkdir(parents=True,exist_ok=True)
        imgs=fetch_images(pid,tmp)
        if not imgs:
            print("[WARN] no image"); shutil.rmtree(tmp,ignore_errors=True); continue

        src=imgs[0]; base=Path(src).stem; dst=OUT_DIR/f"{base}_4k.jpg"
        print("  - upscale 4k…"); upscale_4k(src, str(dst))
        print("  - metadata…");    embed_meta(str(dst), title, kws)
        print("  - ok:", dst)
        done.append(it.get("rowId"))
        shutil.rmtree(tmp,ignore_errors=True)

    if done:
        print("[*] mark used…"); print(post_json(MARK_USED, {"rowIds":done,"log_id":log_id}))

    # upload whole date folder
    try:
        svc=drive()
        root_id=gdrive_ensure_folder(svc, GDRIVE_ROOT_NAME)
        up=gdrive_upload_folder(svc, OUT_DIR, root_id)
        print(f"[*] uploaded {len(up)} files -> Drive/{GDRIVE_ROOT_NAME}/{OUT_DIR.name}")
        if up and len(list(OUT_DIR.glob("*")))>=len(up):
            shutil.rmtree(OUT_DIR); print("[*] removed local date folder")
    except Exception as e:
        print("[WARN] drive upload failed:",e)

if __name__=="__main__":
    main()
