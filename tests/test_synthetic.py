import importlib.util
import sys
from pathlib import Path
import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app.detectors as detectors

spec = importlib.util.spec_from_file_location('synthetic_service', ROOT/'services/synthetic/app.py')
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


def test_window_coverage_and_short_clip():
    assert service.window_starts(1.9) == []
    assert service.window_starts(2.0) == [0.0]
    assert service.window_starts(3.0) == [0.0]
    assert service.window_starts(6.0) == [0.0, 2.0, 4.0]
    assert service.window_starts(15.0) == [0.0, 6.5, 13.0]


def test_invalid_jobs_cannot_access_files():
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        service.locate('../../secret')
    with pytest.raises(HTTPException):
        service.locate('a'*32)


def test_missing_weights_does_not_generate_results(monkeypatch,tmp_path):
    monkeypatch.setattr(service,'WEIGHTS',tmp_path/'non-existent.pth')
    monkeypatch.setattr(service,'_DETECTOR',None)
    with pytest.raises(FileNotFoundError):
        service.get_model()


def test_real_stall_payload_propagated(monkeypatch):
    def handler(request):
        assert request.url.path == '/infer'
        body = request.read().decode()
        assert 'duration_seconds' in body
        return httpx.Response(200,json={
            'status':'complete','model':'STALL / DINOv3 ViT-L/16',
            'score_semantics':'not probability','scored_windows':1,'total_windows':1,'review_windows':1,
            'windows':[{'status':'scored','start_seconds':0.,'end_seconds':2.,
                'real_reference_score':0.07, 'synthetic_anomaly_signal':0.93,'review_flag':True}]
        })
    transport=httpx.MockTransport(handler)
    original=httpx.Client
    monkeypatch.setattr(detectors.httpx,'Client',lambda **kw: original(transport=transport,**kw))
    result=detectors.analyze_synthetic('a'*32,8.)
    assert result['status']=='complete'
    assert result['windows'][0]['real_reference_score']==0.07
    assert result['windows'][0]['synthetic_anomaly_signal']==0.93
    assert '不是' in detectors.summarize_detectors(detectors.disabled('DeepGuard'),detectors.disabled('AASIST'),result) or '不足以' in detectors.summarize_detectors(detectors.disabled('DeepGuard'),detectors.disabled('AASIST'),result)


def test_unavailable_is_not_zero_score(monkeypatch):
    original=httpx.Client
    transport=httpx.MockTransport(lambda _: httpx.Response(503, json={'detail':'no weights'}))
    monkeypatch.setattr(detectors.httpx,'Client',lambda **kw: original(transport=transport,**kw))
    r=detectors.analyze_synthetic('a'*32,12.)
    assert r['status']=='unavailable'
    assert 'windows' not in r


def test_report_separates_synthetic_from_face_score():
    from app.main import report_text
    obj={'created_at':'2026-10-28T00:00:00+00:00','sha256':'abc',
        'media':{'kind':'video'},'c2pa':{'detail':'未發現'},'vlm':{},
        'detectors':{'image_deepfake':detectors.disabled('DeepGuard'),
                     'audio_antispoof':detectors.disabled('AASIST'),
                     'synthetic_video':{'status':'complete','model':'STALL',
                        'warning_threshold':0.75,'windows':[{'start_seconds':0.,'end_seconds':2.,
                         'status':'scored','real_reference_score':0.15,
                         'synthetic_anomaly_signal':0.85,'review_flag':True}]}},'conclusion':'需複核'}
    report=report_text('a'*32,obj)
    assert 'STALL' in report
    assert '0.8500' in report
    assert '不是生成機率' in report or '並非生成機率' in report
