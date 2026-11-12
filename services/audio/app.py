"""AASIST speaker anti-spoof inference on 4.0375-second windows."""
from __future__ import annotations
import json
import math
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path('/data/jobs')
REPO = Path(os.getenv('AASIST_REPO', '/opt/aasist'))
WEIGHTS = Path(os.getenv('AASIST_CHECKPOINT', str(REPO/'models/weights/AASIST.pth')))
app = FastAPI(title='Truth Lab AASIST adapter')
LOCK = threading.Lock()
MODEL = None
DEVICE = None
LOAD_ERROR = None

class AudioRequest(BaseModel):
    job_id: str = Field(pattern=r'^[0-9a-f]{32}$')


def get_model():
    global MODEL, DEVICE, LOAD_ERROR
    with LOCK:
        if MODEL is not None:
            return MODEL
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA GPU unavailable; refusing CPU fallback')
            if not WEIGHTS.is_file() or WEIGHTS.stat().st_size == 0:
                raise FileNotFoundError(f'AASIST checkpoint missing: {WEIGHTS}')
            sys.path.insert(0, str(REPO))
            from models.AASIST import Model
            conf = json.loads((REPO/'config/AASIST.conf').read_text())['model_config']
            model = Model(conf).to('cuda').eval()
            weights = torch.load(WEIGHTS, map_location='cuda', weights_only=True)
            model.load_state_dict(weights, strict=True)
            MODEL, DEVICE, LOAD_ERROR = model, 'cuda', None
            return MODEL
        except Exception as e:
            LOAD_ERROR = f'{type(e).__name__}: {str(e)[:300]}'
            raise


@app.get('/health')
def health():
    return {'status': 'running', 'model_loaded': MODEL is not None,
            'checkpoint_found': WEIGHTS.is_file(), 'last_load_error': LOAD_ERROR,
            'model': 'AASIST'}


@app.post('/warmup')
def warmup():
    try:
        get_model()
        return {'status': 'ready', 'device': DEVICE, 'model': 'AASIST'}
    except Exception as e:
        raise HTTPException(503, f'model unavailable: {type(e).__name__}: {str(e)[:200]}')


def read_audio(path: Path):
    # FFmpeg probes media; strict 30s cap prevents resource abuse even if API metadata is bad.
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-t', '30',
           '-vn', '-ac', '1', '-ar', '16000', '-f', 'f32le', '-acodec', 'pcm_f32le', 'pipe:1']
    r = subprocess.run(cmd, capture_output=True, timeout=55, check=False)
    if r.returncode != 0:
        raise ValueError('FFmpeg audio decoding failed')
    import numpy as np
    waveform = np.frombuffer(r.stdout, dtype='<f4')
    if waveform.size < 1600 or not np.isfinite(waveform).all():
        raise ValueError('Missing, too short or invalid audio')
    return waveform


def chunks(waveform, max_windows=8):
    import numpy as np
    size = 64600
    # Windows are sampled at deterministic offsets, no random cropping.
    last = max(0, len(waveform) - size)
    starts = [int(x) for x in np.linspace(0, last, min(max_windows, max(1, math.ceil(len(waveform)/size))))]
    for start in sorted(set(starts)):
        s = waveform[start:start+size]
        if len(s) < size:
            s = np.tile(s, math.ceil(size/len(s)))[:size]
        yield start/16000, s.copy()


@app.post('/infer')
def infer(req: AudioRequest):
    folder = ROOT / req.job_id
    candidates = [p for p in folder.glob('original.*') if p.suffix.lower() in
                  {'.mp4','.mov','.wav','.mp3','.m4a'}] if folder.is_dir() else []
    if len(candidates) != 1:
        raise HTTPException(404, 'audio media not found')
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(503, f'model unavailable: {type(e).__name__}: {str(e)[:220]}')
    try:
        waveform = read_audio(candidates[0])
    except ValueError as e:
        return {'status': 'insufficient_evidence', 'reason': str(e), 'windows': [], 'model': 'AASIST'}
    import torch
    results = []
    with LOCK, torch.inference_mode():
        for start, samples in chunks(waveform):
            output = model(torch.from_numpy(samples).float().unsqueeze(0).to(DEVICE))[1]
            # ASVspoof label 1 is bonafide; opposite is attack/synthetic.
            bonafide = float(torch.softmax(output, dim=-1)[0,1].item())
            attack_signal = 1.0 - bonafide
            results.append({'seconds': round(start, 2), 'duration_seconds': 4.0375,
                            'attack_score': round(attack_signal, 6)})
    return {'status': 'complete', 'model': 'AASIST', 'dataset': 'ASVspoof2019_LA',
            'score_semantics': 'uncalibrated attack class softmax, not probability of media being fake',
            'windows': results, 'max_attack_score': round(max(x['attack_score'] for x in results), 6)}
