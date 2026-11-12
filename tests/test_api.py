import sys
import io
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('VLM_ENABLED', 'false')
from fastapi.testclient import TestClient
from app.main import app


def test_health():
    with TestClient(app) as client:
        res=client.get('/health')
        assert res.status_code==200
        assert res.json()['version']=='0.3.0'
        assert 'synthetic_enabled' in res.json()


def test_file_validation():
    with TestClient(app) as client:
        res=client.post('/api/analyze',files={'file':('evil.txt',b'hello','text/plain')})
        assert res.status_code==415


def test_job_file_path_rejection():
    with TestClient(app) as client:
        res=client.get('/api/jobs/'+('a'*32)+'/files/../../etc/passwd')
        assert res.status_code in (404,307)
