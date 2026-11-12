import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.detectors import suspicious_intervals, summarize_detectors, disabled
from app.main import report_text


def test_intervals_merge_and_skip_unknown():
    items=[{'seconds':0.5,'status':'scored','score':0.95},
           {'seconds':1.5,'status':'scored','score':0.92},
           {'seconds':2.5,'status':'no_face','score':None},
           {'seconds':3.5,'status':'scored','score':0.79},
           {'seconds':4.5,'status':'scored','score':0.87}]
    observed=suspicious_intervals(items, 0.80, 5.0)
    assert observed==[
        {'start_seconds':0,'end_seconds':2.0,'max_sample_score':0.95,'flagged_samples':2},
        {'start_seconds':4.0,'end_seconds':5.0,'max_sample_score':0.87,'flagged_samples':1}
    ]


def test_no_vision_score_does_not_claim_clean():
    assert suspicious_intervals([{'seconds':1.5,'status':'no_face','score':None}], 0.8,5)==[]
    assert '不代表媒體一定真實' in summarize_detectors(disabled('DeepGuard'), disabled('AASIST'))


def test_report_no_fake_probability():
    doc={'created_at':'2026-10-27T00:00:00+00:00', 'sha256':'abc', 'media':{'kind':'video'},
         'c2pa':{'detail':'缺少憑證'}, 'vlm':{'status':'not_applicable'},
         'detectors':{'image_deepfake':{'status':'complete','model':'ms_eff_gcvit_b0','dataset':'celeb_df_v2',
                                       'total':2,'scored':1,'threshold':0.8,
                                       'frames':[{'seconds':0.5,'status':'scored','score':0.91},
                                                 {'seconds':1.5,'status':'no_face','score':None}],
                                       'flagged_intervals':[{'start_seconds':0,'end_seconds':1,'max_sample_score':0.91}]},
                      'audio_antispoof':disabled('AASIST')},
         'conclusion':'需複核'}
    report=report_text('12345678901234567890123456789012',doc)
    assert '未經校準' in report
    assert '0.50' not in report
    assert '0.91' in report
    assert '未啟用' in report


def test_detector_call_propagates_real_scores(monkeypatch):
    import httpx
    import app.detectors as d
    def handler(request):
        assert request.url.path == '/infer'
        assert 'job_id' in request.read().decode()
        return httpx.Response(200, json={
            'status':'complete', 'model':'ms_eff_gcvit_b0', 'dataset':'celeb_df_v2',
            'scored':1, 'total':2,
            'frames':[{'status':'scored','seconds':0.5,'filename':'detect_frame_000.jpg','score':0.91},
                      {'status':'no_face','seconds':1.5,'filename':'detect_frame_001.jpg','score':None}]})
    transport=httpx.MockTransport(handler)
    original=httpx.Client
    monkeypatch.setattr(d.httpx,'Client',lambda **kw: original(transport=transport,**kw))
    result=d.analyze_vision('a'*32,[{'seconds':0.5,'filename':'detect_frame_000.jpg'}],2)
    assert result['flagged_samples']==1
    assert result['flagged_intervals'][0]['max_sample_score']==0.91
    assert result['frames'][1]['score'] is None


def test_detector_network_failure_means_unavailable(monkeypatch):
    import httpx
    import app.detectors as d
    def handler(request):
        raise httpx.ConnectError('test network down')
    transport=httpx.MockTransport(handler)
    original=httpx.Client
    monkeypatch.setattr(d.httpx,'Client',lambda **kw: original(transport=transport,**kw))
    result=d.analyze_vision('a'*32,[{'seconds':0.5,'filename':'detect_frame_000.jpg'}],2)
    assert result['status']=='unavailable'
    assert 'score' not in result


def test_conclusion_distinguishes_flagged_and_unflagged_samples():
    audio = disabled('AASIST')
    assert '有疑似造假或生成的訊號' in summarize_detectors(
        {'status': 'complete', 'scored': 2, 'flagged_samples': 1}, audio)
    result = summarize_detectors({'status': 'complete', 'scored': 2, 'flagged_samples': 0}, audio)
    assert '未發現超過門檻' in result
    assert '取樣範圍' in result


def test_conclusion_synthetic_and_audio_scope():
    vision = disabled('DeepGuard')
    audio = {'status': 'complete', 'windows': [{'attack_score': 0.99}]}
    assert '無法判定真偽' in summarize_detectors(vision, audio)
    result = summarize_detectors(vision, audio, {'status': 'complete', 'scored_windows': 1, 'review_windows': 1})
    assert '有疑似造假或生成的訊號' in result
    assert '聲音：已完成分析' in result
