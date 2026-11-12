#!/usr/bin/env python3
"""Standard-library smoke test for the AI Truth Lab synthetic video detector.

Usage: python3 scripts/smoke_synthetic.py /path/to/clip.mp4 http://localhost:8088
"""
import json
import mimetypes
import sys
import time
import uuid
import urllib.request
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 scripts/smoke_synthetic.py <video.mp4> [base_url]', file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    base = (sys.argv[2] if len(sys.argv)>2 else 'http://localhost:8088').rstrip('/')
    if not path.is_file() or path.suffix.lower() not in ('.mp4','.mov'):
        print('Expected existing .mp4 or .mov file',file=sys.stderr)
        return 2
    health = json.load(urllib.request.urlopen(base+'/health',timeout=15))
    print('API:',health)
    if health.get('version') != '0.3.0' or not health.get('synthetic_enabled'):
        print('FAIL: API v0.3.0 and SYNTHETIC_ENABLED=true required')
        return 1
    if path.stat().st_size > 50*1024*1024:
        print('FAIL: >50MB sample, use a shorter clip')
        return 2
    boundary='----truthlab'+uuid.uuid4().hex
    head=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
          'Content-Type: video/mp4\r\n\r\n').encode()
    body=head+path.read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
    req=urllib.request.Request(base+'/api/analyze',data=body,headers={'Content-Type':f'multipart/form-data; boundary={boundary}'},method='POST')
    job=json.load(urllib.request.urlopen(req,timeout=45))['job_id']
    print('job_id:',job)
    for _ in range(200):
        data=json.load(urllib.request.urlopen(base+'/api/jobs/'+job,timeout=20))
        if data.get('state') == 'failed':
            print('FAIL:',data.get('error'))
            return 1
        if data.get('state') == 'complete':
            output=data['report']['detectors']['synthetic_video']
            print('STALL:',json.dumps(output,ensure_ascii=False,indent=2))
            if output.get('status') == 'complete' and output.get('scored_windows',0)>0:
                print('PASS: STALL returned actual window scores (NOT an accuracy validation)')
                return 0
            print('FAIL: STALL did not return usable scores; check warmup and service logs')
            return 1
        if _ % 10 == 0:
            print(data.get('state'),data.get('progress'),data.get('step'))
        time.sleep(3)
    print('FAIL: timed out waiting for API job')
    return 1


if __name__=='__main__':
    sys.exit(main())
