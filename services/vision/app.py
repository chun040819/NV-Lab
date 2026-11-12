"""True DeepGuard inference; reports missing face/model errors as unknown, not 0.5."""
from __future__ import annotations
import math
import os
import re
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path('/data/jobs')
MODEL_NAME = os.getenv('VISION_MODEL', 'ms_eff_gcvit_b0')
DATASET = os.getenv('VISION_DATASET', 'celeb_df_v2')
app = FastAPI(title='Truth Lab DeepGuard adapter')
LOCK = threading.Lock()
PRED = None
LOAD_ERROR = None


class VisionRequest(BaseModel):
    job_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    frames: list[dict] = Field(min_length=1, max_length=25)


def get_predictor():
    global PRED, LOAD_ERROR
    with LOCK:
        if PRED is not None:
            return PRED
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA GPU unavailable; refusing CPU fallback')
            from inference.image_predictor import ImagePredictor
            predictor = ImagePredictor(margin_ratio=0.2, conf_thres=0.5,
                                       min_face_ratio=0.01, model_name=MODEL_NAME,
                                       dataset=DATASET)
            PRED = predictor
            LOAD_ERROR = None
            return PRED
        except Exception as e:
            LOAD_ERROR = f'{type(e).__name__}: {str(e)[:350]}'
            raise


@app.get('/health')
def health():
    return {'status': 'running', 'model_loaded': PRED is not None,
            'last_load_error': LOAD_ERROR, 'model': MODEL_NAME, 'dataset': DATASET}


@app.post('/warmup')
def warmup():
    try:
        p = get_predictor()
        return {'status': 'ready', 'device': p.device, 'model': MODEL_NAME, 'dataset': DATASET}
    except Exception as e:
        raise HTTPException(503, f'model is not ready: {type(e).__name__}: {str(e)[:280]}')


def score_frame(predictor, path):
    import torch
    from deepguard.data import get_test_transforms
    # Upstream predict_img() returns 0.5 on failed detection; NEVER use that method.
    crop = predictor._preprocess_img(str(path))
    if crop is None:
        return {'status': 'no_face', 'score': None, 'reason': '未偵測到可分析的人臉／影格讀取失敗'}
    transform = get_test_transforms(img_size=predictor.img_size, tta_hflip=0)
    image = transform(image=crop)['image'].unsqueeze(0).to(predictor.device)
    with torch.inference_mode():
        output = predictor.model(image)
        score = float(torch.sigmoid(output.view(-1)[0]).item())
    if not math.isfinite(score):
        raise ValueError('non-finite model output')
    return {'status': 'scored', 'score': round(score, 6)}


@app.post('/infer')
def infer(req: VisionRequest):
    folder = ROOT / req.job_id
    if not folder.is_dir():
        raise HTTPException(404, 'job not found')
    try:
        p = get_predictor()
    except Exception as e:
        raise HTTPException(503, f'model unavailable: {type(e).__name__}: {str(e)[:200]}')
    results = []
    with LOCK:
        for frame in req.frames:
            name = frame.get('filename')
            seconds = frame.get('seconds')
            if not isinstance(name, str) or not re.fullmatch(r'detect_frame_\d{3}\.jpg|preview\.jpg', name):
                raise HTTPException(400, 'invalid frame name')
            if not isinstance(seconds, (int, float)) and seconds is not None:
                raise HTTPException(400, 'invalid frame time')
            path = folder / name
            if not path.is_file():
                raise HTTPException(404, f'missing frame: {name}')
            try:
                finding = score_frame(p, path)
            except Exception as e:
                finding = {'status': 'error', 'score': None, 'reason': f'{type(e).__name__}: {str(e)[:150]}'}
            results.append({'filename': name, 'seconds': seconds, **finding})
    return {'status': 'complete', 'model': MODEL_NAME, 'dataset': DATASET,
            'score_semantics': 'uncalibrated DeepGuard sigmoid; larger indicates manipulation signal',
            'frames': results, 'total': len(results),
            'scored': sum(x['status'] == 'scored' for x in results)}
