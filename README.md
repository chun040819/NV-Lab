# AI Truth Lab — DGX Spark 全 Docker MVP v0.2

**用途：2026 年 11 月底記者會的教學／互動鑑識示範，不是正式鑑識報告系統。** 影像／聲音模型的分數不是經校準的造假機率，沒有 C2PA 不等於偽造，也不能僅靠 AI 分數宣稱某事件為假。

## 第二階段新增了什麼

- 保留 v0.1 的 FastAPI、Web Dashboard、Ollama/Gemma 3、C2PA、SHA-256、FFmpeg、報告與 QR Code。
- 新增 `vision-detector`，使用 `DeepGuard 1.1.1`、YOLO 人臉偵測與 `ms_eff_gcvit_b0` 預訓練模型。每秒取樣一幀（最長 20 秒），輸出實際模型分數；沒有人臉、沒有模型權重、服務錯誤時回報 `no_face` / `unavailable`，**不產生假的 0.5 分數**。模型首次載入需要下載多個權重。
- 新增 `audio-detector`，使用原始 `clovaai/aasist` 公開的預訓練模型，解碼成 16 kHz 單聲道，對最長 30 秒的音訊／含聲影片按 64600 樣本取多個窗。
- Dashboard 顯示按取樣點模型分數標記的區間、逐影格結果與音訊模型摘要；JSON/Markdown 報告記錄模型名稱與資料集。
- 兩個偵測服務均只連入 Compose 的 `internal` network，不開放宿主機 Port；以 Compose profiles 選擇性啟動，避免壞掉時拖垮 v0.1。

## 重要前提

這份專案已通過本地 Python 語法、unit tests 與前端語法檢查，**尚未在實體 DGX Spark 進行 GPU 容器 Build / 實際權重下載 / 真實模型推論**。NVIDIA NGC `pytorch:25.11-py3` 在當年具 ARM64 / GB10 GPU 支援，這裡刻意採用 CUDA 13.0 世代的保守預設，以降低驅動相容風險。請先用 `nvidia-smi` 驗證主機 GPU Driver；若版本已滿足 CUDA 13.4，可自行改用新版 `26.09-py3` 進行相容性測試，不要直接在記者會當天更換。

深度學習模型需要從上游專案／下載伺服器取得權重，第一次使用前必須可上網；預載完成後才能做離線測試。NGC Docker 基底容量較大；新增兩個服務可能佔用數十 GB 磁碟。部署前確認容量。

## 1. 備份第一階段部署

在 DGX Spark 上：

```bash
cd ~
cp -a ai-truth-lab "ai-truth-lab-backup-$(date +%Y%m%d)"
```

將 `ai-truth-lab-stage2.zip` 傳到 DGX Spark 家目錄，然後：

```bash
cd ~
unzip -o ai-truth-lab-stage2.zip -d truthlab-stage2-update
cp -a truthlab-stage2-update/ai-truth-lab/. ai-truth-lab/
cd ai-truth-lab
# 檢查原本的 .env 與 data/jobs 都還在
ls -la .env data
```

資料檔、`data/jobs` 不會被升級 ZIP 刪掉；若原本有自己修改過的程式或前端，請先比較差異再覆寫。

## 2. 先確保 v0.1 功能不受影響

`.env` 新增參數（若未填，Compose 預設全部關閉）：

```dotenv
VISION_ENABLED=false
AUDIO_ENABLED=false
VISION_MODEL=ms_eff_gcvit_b0
VISION_DATASET=celeb_df_v2
VISION_THRESHOLD=0.80
MAX_AUDIO_SECONDS=30
DETECTOR_TIMEOUT_SECONDS=240
PYTORCH_BASE=nvcr.io/nvidia/pytorch:25.11-py3
```

```bash
docker compose build api
docker compose up -d api ollama
curl -s http://localhost:8088/health
# 預期 version 0.2.0，vision_enabled/audio_enabled=false
```

再次上傳一張圖片，確認 C2PA、VLM、報告仍然正常。

## 3. 建置並啟用 DeepGuard（建議先做）

```bash
# 先單獨驗證 PyTorch GPU 容器
# 注意：NGC 映像很大，首次下載需要時間／磁碟空間。
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 \
  python -c "import torch; print(torch.__version__); print('cuda:',torch.cuda.is_available()); print('gpu:',torch.cuda.get_device_name(0))"

# 建置與啟動獨立模型服務
docker compose --profile vision build vision-detector
docker compose --profile vision up -d vision-detector

docker compose --profile vision ps
# 暖機：首次下載 YOLO face / EfficientNet backbone / DeepGuard 權重
docker compose --profile vision exec -T vision-detector \
  python -c "import urllib.request; r=urllib.request.Request('http://localhost:8091/warmup',data=b'{}',method='POST',headers={'Content-Type':'application/json'}); print(urllib.request.urlopen(r,timeout=240).read().decode())"
```

暖機成功會回傳 `status=ready`、`device=cuda:0`，且模型已載入。若此步失敗，先讀取：

```bash
docker compose --profile vision logs --tail=160 vision-detector
```

成功後把 `.env` 設定為：

```dotenv
VISION_ENABLED=true
```

然後：

```bash
docker compose up -d --force-recreate api
curl -s http://localhost:8088/health
```

用已授權且已知來源的真人照片及換臉照片測試；對影片要求 `detectors.image_deepfake.frames` 不為空，且有非 `null` 的模型分數。影像中沒有人臉時應顯示 `no_face`，不是 0.5。

## 4. 建置並啟用 AASIST（影像成功後才開始）

```bash
docker compose --profile audio build audio-detector
docker compose --profile audio up -d audio-detector

docker compose --profile audio exec -T audio-detector \
  python -c "import urllib.request; r=urllib.request.Request('http://localhost:8092/warmup',data=b'{}',method='POST',headers={'Content-Type':'application/json'}); print(urllib.request.urlopen(r,timeout=240).read().decode())"
```

啟動成功後將 `.env` 設定 `AUDIO_ENABLED=true`，重建 API 的執行環境：

```bash
docker compose up -d --force-recreate api
```

分別上傳已知來源的真人錄音與授權合成語音，確認每段音訊窗都有**實際計算**的 `attack_score`。**AASIST 的類別索引 1 是 bona fide**，此專案計算 `1 - softmax(logits)[1]` 作為反偽造訊號；它不是真偽的經校準機率。AASIST 是 ASVspoof 2019 LA 訓練模型，對新型合成語音、台灣語音、會場收音的誤差不能假設很小。

## 5. 真正的驗收方式

四種測試素材：

| 樣本 | 預期系統行為 |
|---|---|
| 已授權真實人臉照片 | 計分成功，記錄分數，不能宣稱「保證真」 |
| 已知人工換臉照片／影片 | 計分成功；如分數超過展示門檻，顯示待複核影格／區間 |
| 無人臉的風景照片 | 回傳 `no_face`/`insufficient_evidence`，不能標示成真實 |
| 已知授權的真人與合成聲音 | AASIST 各產出取樣窗結果，明確標注限制 |

另需對無聲影片、損壞檔案、服務沒有啟動、模型權重下載失敗、網路斷線等情況測試。應使用至少 20 對素材建立混淆矩陣，並針對影格與音訊窗分析不同門檻；`VISION_THRESHOLD=0.80` 僅是暫定展示操作門檻，**不代表模型已校準**。如果真實素材的錯誤率不符合展示需求，應降級成 C2PA 與可觀察影格對照，而不是偽造結果。

## 6. 關於時間軸的準確意義

影片每秒取樣 1 幀（預設 0.5s、1.5s ...），只在該幀有偵測到人臉時運行人臉操作模型。分數 >= `VISION_THRESHOLD` 的取樣點，在圖上顯示 `t ± 0.5 秒` 作為「需人工複核範圍」並合併相鄰區間。**這不代表真的改動從哪一毫秒開始或結束**，不能稱為精確的竄改時間定位。若要達到研究級時序定位，需要逐格密集推論、時間一致性模型和有真實逐格標註的測試資料。

## 7. 故障排除與降級

```bash
docker compose --profile vision --profile audio ps
docker compose --profile vision logs --tail=200 vision-detector
docker compose --profile audio logs --tail=200 audio-detector
docker compose logs --tail=200 api
nvidia-smi
```

- 如果 `model unavailable`：確認模型暖機是否成功、網路可下載權重、API 與對應 profile 是否啟動。
- 如果 `no_face`：不是模型判定真人，代表沒有可檢測人臉／影格無法分析。
- 如果下載或 NGC Image 不相容：先停用 `.env` 的偵測服務，在不影響 C2PA/VLM 的條件下排查；不要在正式演示時倉促升級主機 CUDA。
- 如果模型服務不穩定：設 `VISION_ENABLED=false` 或 `AUDIO_ENABLED=false`，執行 `docker compose up -d --force-recreate api`，原 v0.1 核心應繼續可用。
- 服務只跑在 Docker；GPU 使用 Compose `gpus: all`，API 只經由內部 Compose DNS 與 detector 溝通，沒有將模型 API Port 暴露到外部。
- QR Code 報告網址沒有身份驗證或期限，請只在隔離的記者會內網展示，**不要將此 MVP 直接暴露到公網**。正式開放給記者自由上傳前還要做登入、限流、清理期限、病毒掃描與檔案安全措施。

## API

- `GET /health`
- `POST /api/analyze` (`multipart/form-data`, `file`)
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/files/report.json`
- `GET /api/jobs/{job_id}/files/report.md`
- `GET /r/{job_id}`
- `GET /api/jobs/{job_id}/qr`

專用模型服務僅在 Docker network 使用 `POST /infer`、`POST /warmup`、`GET /health`。

## 上游來源

- NVIDIA PyTorch NGC：`nvcr.io/nvidia/pytorch:25.11-py3`
- DeepGuard：<https://github.com/HanMoonSub/DeepGuard> （PyPI `deepguard==1.1.1`）
- AASIST：<https://github.com/clovaai/aasist> （原作者的程式與預訓練權重）
- c2patool：<https://github.com/contentauth/c2patool>

MIT 與其他上游授權及模型權重的條件，必須在正式對外使用之前逐項審核。

---

## v0.3 — 全生成影片偵測

新增 `synthetic-video-detector` (Docker Compose profile `synthetic`)；詳細升級、模型授權、官方 DINOv3 權重取得、GPU 暖機與驗收見 [UPGRADE-v0.3.md](UPGRADE-v0.3.md)。`SYNTHETIC_ENABLED=false` 為安全預設，未暖機前不會提交 STALL 推論。STALL 分數不能用作任何真偽或詐騙定論。
