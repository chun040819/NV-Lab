"""Single-video STALL inference API. Honest, non-probabilistic scores.

STALL's final score is a percentile relative to VATEX *real* videos;
HIGHER = more real-like. 1-score is NOT P(AI generated).
"""
from __future__ import annotations
import math
import os
import re
import sys
import threading
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

STALL_DIR = Path(os.getenv("STALL_DIR", "/opt/stall"))
WEIGHTS = Path(os.getenv("DINO_V3_WEIGHTS", "/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"))
PARAMS = Path(os.getenv("STALL_PARAMS", str(STALL_DIR / "precomputed/stall_params_vatex_dino_v3.npz")))
FRAME_BATCH = max(1, min(16, int(os.getenv("STALL_FRAME_BATCH", "4"))))
MAX_WINDOWS = max(1, min(5, int(os.getenv("STALL_MAX_WINDOWS", "3"))))
WARNING_THRESHOLD = float(os.getenv("STALL_WARNING_THRESHOLD", "0.75"))
FPS = 8
WINDOW_SECONDS = 2
LOCK = threading.RLock()
_DETECTOR = None
_DEVICE = None
app = FastAPI(title="Truth Lab Synthetic Video — STALL", version="0.3.0")

class InferRequest(BaseModel):
    job_id: str = Field(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$")
    duration_seconds: float | None = Field(default=None, ge=0, le=120)


def window_starts(duration: float, window_seconds: float = WINDOW_SECONDS, max_windows: int = MAX_WINDOWS) -> list[float]:
    """Sample up to three disjoint positions (start/middle/end); do not claim localization."""
    if not math.isfinite(duration) or duration < window_seconds:
        return []
    last = max(0., duration - window_seconds)
    if last < .15: return [0.]
    n = min(max_windows, max(1, int(math.floor(duration / window_seconds))))
    if n == 1: return [0.]
    return sorted(set(round(last * i / (n-1), 3) for i in range(n)))


def locate(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(422,"bad job ID")
    root = Path('/data/jobs') / job_id
    choices = [p for ext in ('mp4','mov') if (p:=root/f'original.{ext}').is_file()]
    if len(choices) != 1:
        raise HTTPException(404, 'video for job not found')
    return choices[0]


def get_model():
    global _DETECTOR, _DEVICE
    with LOCK:
        if _DETECTOR is not None: return _DETECTOR
        if not WEIGHTS.is_file():
            raise FileNotFoundError('DINOv3 weights absent: download the authorized ViT-L/16 file from Meta and mount to /models')
        if not PARAMS.is_file():
            raise FileNotFoundError('STALL VATEX calibration parameters absent')
        import numpy as np
        import torch
        sys.path.insert(0, str(STALL_DIR / 'src'))
        from stall import STALL
        _DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if _DEVICE == 'cpu':
            raise RuntimeError('CUDA unavailable; do not perform a slow CPU fallback silently')
        with np.load(str(PARAMS), allow_pickle=False) as data:
            params = {k:data[k] for k in data.files}
        _DETECTOR = STALL(device=_DEVICE, data_dict=params,
                         dino_repo=os.getenv('DINO_V3_REPO_DIR', '/opt/dinov3'),
                         dino_weights=str(WEIGHTS))
        return _DETECTOR


def sample_frame_window(cap, fps: float, start: float):
    import cv2
    import numpy as np
    frames=[]
    indices=[]
    for i in range(FPS * WINDOW_SECONDS):
        second=start + (i+.5)/FPS
        idx=int(round(second*fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame=cap.read()
        if not ok or frame is None:
            return None
        frames.append(frame)
        indices.append(idx)
    # A 1 fps/still image clip does not provide independent 8 fps motion samples.
    if len(set(indices)) < 8:
        return None
    return np.stack(frames)


def score_one(detector, frames):
    import numpy as np
    embeddings=detector.frames_to_embeddings([frames], batch_size=FRAME_BATCH)
    result=detector._scores_from_embs(embeddings)  # API of pinned STALL release
    spatial=float(result['spat_percentile'][0])
    temporal=float(result['temp_percentile'][0])
    real_like=float(result['final_score'][0])
    if not all(math.isfinite(x) and 0<=x<=1 for x in (spatial,temporal,real_like)):
        raise ValueError('non-finite or invalid STALL score')
    return spatial,temporal,real_like


@app.get('/health')
def health():
    return {'status':'ok','model_loaded':_DETECTOR is not None,
            'weights_present':WEIGHTS.is_file(), 'params_present':PARAMS.is_file(),
            'device':_DEVICE, 'score_direction':'higher_is_real_like'}

@app.post('/warmup')
def warmup():
    try:
        get_model()
    except Exception as exc:
        raise HTTPException(503, f'Unable to load STALL: {type(exc).__name__}: {exc}') from exc
    return {'status':'ready','device':_DEVICE, 'model':'STALL / DINOv3 ViT-L/16'}

@app.post('/infer')
def infer(req: InferRequest):
    video=locate(req.job_id)
    with LOCK:  # shared DINOv3 model; serialize concurrent work
        try:
            model=get_model()
        except Exception as exc:
            raise HTTPException(503,f'STALL unavailable: {type(exc).__name__}: {exc}') from exc
        import cv2
        started=perf_counter()
        cap=cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise HTTPException(422,'Video cannot be decoded')
        try:
            fps=float(cap.get(cv2.CAP_PROP_FPS))
            count=float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps<=0 or not math.isfinite(count) or count<=0:
                return {'status':'insufficient_evidence','model':'STALL / DINOv3 ViT-L/16',
                        'reason':'影片幀率或影格數無法確認','windows':[]}
            duration=count/fps
            duration=min(duration, req.duration_seconds) if req.duration_seconds else duration
            if duration>120:
                raise HTTPException(422,'Video too long')
            starts=window_starts(duration)
            if not starts:
                return {'status':'insufficient_evidence','model':'STALL / DINOv3 ViT-L/16',
                        'reason':'需要至少 2 秒可分析影片','windows':[]}
            windows=[]
            for start in starts:
                frames=sample_frame_window(cap,fps,start)
                if frames is None:
                    windows.append({'start_seconds':start,'end_seconds':round(start+WINDOW_SECONDS,3),
                                    'status':'insufficient_evidence','reason':'影格不足或無法穩定解碼'})
                    continue
                try:
                    sp,tm,real=score_one(model,frames)
                except Exception as exc:
                    windows.append({'start_seconds':start,'end_seconds':round(start+WINDOW_SECONDS,3),
                                    'status':'unavailable','reason':f'{type(exc).__name__}: {exc}'})
                    continue
                synthetic_signal=round(1-real,6)
                windows.append({'start_seconds':start,'end_seconds':round(start+WINDOW_SECONDS,3),
                                'status':'scored','spatial_real_percentile':round(sp,6),
                                'temporal_real_percentile':round(tm,6),
                                'real_reference_score':round(real,6),
                                'synthetic_anomaly_signal':synthetic_signal,
                                'review_flag':synthetic_signal>=WARNING_THRESHOLD})
            scored=[x for x in windows if x['status']=='scored']
            return {'status':'complete' if scored else 'insufficient_evidence',
                    'model':'STALL / DINOv3 ViT-L/16','device':_DEVICE,
                    'score_semantics':'STALL 校準集相對百分位：越高越接近校準真實影片；1-分數是異常訊號，不是 AI 生成機率',
                    'params':'VATEX / official STALL','window_seconds':WINDOW_SECONDS,
                    'sample_fps':FPS,'warning_threshold':WARNING_THRESHOLD,
                    'threshold_note':'暫定展示門檻，未在使用者資料上校準',
                    'windows':windows,'scored_windows':len(scored),'total_windows':len(windows),
                    'review_windows':sum(x['review_flag'] for x in scored),
                    'mean_real_reference_score':round(sum(x['real_reference_score'] for x in scored)/len(scored),6) if scored else None,
                    'elapsed_seconds':round(perf_counter()-started,3),
                    'scope_note':'偵測整體生成異常；視窗不是精確的修改起訖位置'}
        finally:
            cap.release()
