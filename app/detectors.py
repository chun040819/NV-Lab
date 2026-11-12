"""Evidence-preserving detector orchestration; never substitutes fabricated scores."""
from __future__ import annotations
import os
from typing import Any

import httpx

VISION_URL = os.getenv('VISION_URL', 'http://vision-detector:8091').rstrip('/')
AUDIO_URL = os.getenv('AUDIO_URL', 'http://audio-detector:8092').rstrip('/')
SYNTHETIC_URL = os.getenv('SYNTHETIC_URL', 'http://synthetic-video-detector:8093').rstrip('/')
SYNTHETIC_TIMEOUT = float(os.getenv('SYNTHETIC_TIMEOUT_SECONDS', '480'))
TIMEOUT = float(os.getenv('DETECTOR_TIMEOUT_SECONDS', '240'))
THRESHOLD = float(os.getenv('VISION_THRESHOLD', '0.80'))


def disabled(model: str) -> dict:
    return {'status': 'disabled', 'model': model, 'reason': '服務未啟用，不提供偵測結果'}


def unavailable(model: str, err: BaseException) -> dict:
    return {'status': 'unavailable', 'model': model,
            'reason': f'模型服務目前不可用（{type(err).__name__}）；不可推論為未發現造假'}


def suspicious_intervals(frames: list[dict], threshold: float, duration: float | None,
                         radius: float = 0.5) -> list[dict]:
    """Threshold-based sampling intervals, NOT a localization ground truth."""
    ranges = []
    if not 0 < threshold < 1:
        raise ValueError('threshold must be between 0 and 1')
    for row in frames:
        t, score = row.get('seconds'), row.get('score')
        if row.get('status') != 'scored' or t is None or score is None or score < threshold:
            continue
        s, e = max(0, t-radius), t+radius
        if duration is not None:
            e = min(duration, e)
        if e > s:
            ranges.append([round(s, 2), round(e, 2), score])
    ranges.sort(key=lambda x:x[0])
    merged = []
    for s,e,score in ranges:
        if merged and s <= merged[-1]['end_seconds'] + 1e-6:
            merged[-1]['end_seconds'] = max(merged[-1]['end_seconds'],e)
            merged[-1]['max_sample_score'] = max(merged[-1]['max_sample_score'],score)
            merged[-1]['flagged_samples'] += 1
        else:
            merged.append({'start_seconds':s,'end_seconds':e,
                           'max_sample_score':score,'flagged_samples':1})
    for row in merged:
        row['max_sample_score'] = round(row['max_sample_score'],6)
    return merged


def analyze_vision(job_id: str, frames: list[dict], duration: float | None) -> dict:
    if not frames:
        return {'status': 'insufficient_evidence', 'model': 'DeepGuard', 'reason': '沒有可用影格'}
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp=client.post(f'{VISION_URL}/infer',json={'job_id':job_id,'frames':frames})
            resp.raise_for_status()
            result=resp.json()
        if not isinstance(result.get('frames'),list):
            raise ValueError('invalid detector response')
        result['threshold'] = THRESHOLD
        result['threshold_note'] = '展示用未校準門檻，非真假機率／標準鑑識門檻'
        result['flagged_intervals'] = suspicious_intervals(result['frames'], THRESHOLD, duration)
        result['flagged_samples'] = sum(1 for x in result['frames']
                                        if x.get('status')=='scored' and x.get('score') is not None
                                        and x['score'] >= THRESHOLD)
        if not result.get('scored'):
            result['status']='insufficient_evidence'
        return result
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        return unavailable('DeepGuard', exc)


def analyze_audio(job_id: str) -> dict:
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f'{AUDIO_URL}/infer',json={'job_id':job_id})
            resp.raise_for_status()
            result = resp.json()
        if not isinstance(result,dict) or result.get('status') not in ('complete','insufficient_evidence'):
            raise ValueError('invalid audio detector response')
        return result
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        return unavailable('AASIST', exc)


def analyze_synthetic(job_id: str, duration_seconds: float | None = None) -> dict:
    """Run STALL on original video, never on DeepGuard's face crops."""
    try:
        with httpx.Client(timeout=SYNTHETIC_TIMEOUT) as client:
            response = client.post(f'{SYNTHETIC_URL}/infer', json={
                'job_id': job_id, 'duration_seconds': duration_seconds})
            response.raise_for_status()
            result = response.json()
        if not isinstance(result, dict) or result.get('status') not in ('complete', 'insufficient_evidence'):
            raise ValueError('invalid STALL status')
        if not isinstance(result.get('windows'), list):
            raise ValueError('invalid STALL windows')
        for window in result['windows']:
            if window.get('status') == 'scored':
                score = window.get('real_reference_score')
                if not isinstance(score, (float, int)) or not 0 <= score <= 1:
                    raise ValueError('invalid STALL real reference score')
        return result
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        return unavailable('STALL / DINOv3', exc)


def summarize_detectors(vision: dict, audio: dict, synthetic: dict | None = None) -> str:
    """Describe observed signals without treating uncalibrated scores as facts."""
    findings = []
    flagged = False
    if vision.get('status') == 'complete' and vision.get('scored', 0) > 0:
        count = vision.get('flagged_samples', 0)
        flagged |= count > 0
        findings.append('人臉影像：' + ('偵測到疑似合成或操弄訊號' if count else '已分析的人臉影格未達疑似操弄門檻'))
    if synthetic and synthetic.get('status') == 'complete' and synthetic.get('scored_windows', 0) > 0:
        count = synthetic.get('review_windows', 0)
        flagged |= count > 0
        findings.append('全片生成：' + ('偵測到異常訊號' if count else '已分析的影片視窗未達異常門檻'))
    if audio.get('status') == 'complete' and audio.get('windows'):
        findings.append('聲音：已完成分析，分數尚無校準門檻，無法判定真偽')
    if flagged:
        verdict = '初步判斷：有疑似造假或生成的訊號，建議人工複核。'
    elif any(x.startswith(('人臉影像：', '全片生成：')) for x in findings):
        verdict = '初步判斷：已分析的範圍未發現超過門檻的疑似造假訊號。'
    else:
        verdict = '目前無法判定真偽；模型未啟用或沒有可分析的影像，不代表媒體一定真實。'
    if findings:
        verdict += ' ' + '；'.join(findings) + '。'
    if flagged or any(x.startswith(('人臉影像：', '全片生成：')) for x in findings):
        verdict += ' 此結果僅涵蓋取樣範圍，門檻未經目標資料校準，不能確定媒體真偽。'
    return verdict
