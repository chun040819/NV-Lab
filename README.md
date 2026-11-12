# AI Truth Lab

AI Truth Lab 是媒體鑑識示範工具，可上傳圖片、影片或音訊，查看來源憑證、模型分析結果及可下載的報告。專案以 Docker Compose 執行，主要針對具備 NVIDIA GPU 的環境設計。

模型結果是輔助判讀的訊號，並非經校準的真假機率或正式鑑識結論。未發現異常不代表媒體一定真實，缺少 C2PA 憑證也不代表造假。

## 快速上手

### 1. 準備環境

- 安裝 Docker Engine、Docker Compose 與 NVIDIA Container Toolkit，並確認 Docker 能使用 NVIDIA GPU。
- 執行 `nvidia-smi` 確認主機可辨識 GPU。專案預設使用 NVIDIA NGC PyTorch `25.11-py3`，實際相容性仍需以主機驅動、映像建置及模型暖機結果確認。
- 首次建置與下載模型需要網路連線及足夠磁碟空間；GPU 映像與權重可能需要數十 GB。

### 2. 下載並設定

```bash
git clone https://github.com/chun040819/NV-Lab.git
cd NV-Lab
cp .env.example .env
```

開啟 `.env`，將 `PUBLIC_BASE_URL` 設為其他裝置可以連線的網址，例如 `http://192.168.1.100:8088`；這個網址用於報告 QR Code。僅在本機使用時可設為 `http://localhost:8088`。

### 3. 啟動並下載視覺語言模型

```bash
docker compose up -d --build
# 第一次需要下載模型；模型名稱須與 .env 的 OLLAMA_MODEL 相同
docker compose exec ollama ollama pull gemma3:4b
curl http://localhost:8088/health
```

開啟瀏覽器前往 **http://localhost:8088**，上傳檔案並等待分析完成。支援 JPG、JPEG、PNG、MP4、MOV、WAV、MP3、M4A；預設上限為 50 MB、影片 20 秒、音訊 30 秒，可在 `.env` 調整。

**預設只啟動 API 和 Ollama，沒有啟用人臉、聲音或全片生成偵測模型。** 因此僅完成上述步驟時，不會得到這些模型的偵測結論。若要偵測人臉或聲音，請繼續下面的步驟。

## 啟用人臉影像與聲音偵測

先建置並啟動服務：

```bash
docker compose --profile vision --profile audio up -d --build vision-detector audio-detector
```

暖機並確認兩個命令都成功；DeepGuard 首次暖機會下載權重：

```bash
docker compose --profile vision exec -T vision-detector \
  python -c "import urllib.request; r=urllib.request.Request('http://localhost:8091/warmup',data=b'{}',method='POST'); print(urllib.request.urlopen(r,timeout=240).read().decode())"
docker compose --profile audio exec -T audio-detector \
  python -c "import urllib.request; r=urllib.request.Request('http://localhost:8092/warmup',data=b'{}',method='POST'); print(urllib.request.urlopen(r,timeout=240).read().decode())"
```

暖機成功後，在 `.env` 修改：

```dotenv
VISION_ENABLED=true
AUDIO_ENABLED=true
```

重新建立 API，讓設定生效：

```bash
docker compose up -d --force-recreate api
curl http://localhost:8088/health
```

重新上傳素材即可查看對應模型結果。人臉影像模型只分析偵測到人臉的取樣影格；聲音模型的分數目前沒有經驗證的判定門檻。服務未啟動、無人臉或音訊無法解碼時，不會產生可靠的模型結論。

## 啟用全片生成影片偵測

STALL 需要另外取得授權的 DINOv3 權重，放到 `models/dinov3/`，並建置、暖機 `synthetic` profile 後才能啟用。詳細步驟、權重檔名與授權限制請見 [UPGRADE-v0.3.md](UPGRADE-v0.3.md)。STALL 的分數是相對真實影片校準集的訊號，不是 AI 生成機率。

## 結果與限制

- 人臉或全片模型超過展示門檻時，結果會提示疑似訊號及人工複核；未超過時，只代表已分析的取樣範圍沒有超過門檻。
- 影像的預設 `VISION_THRESHOLD=0.80` 與 STALL 展示門檻尚未在使用者素材上校準，不能據此確定媒體真偽。
- 影片標記區間由取樣點擴展而成，不是精確的竄改起訖時間。建議使用已知來源的真實與合成素材測試，評估誤判情況。
- 報告與上傳資料儲存在 `data/`；Ollama 與部分模型權重使用 Docker volumes。

## 常用指令與故障排除

```bash
# 查看服務狀態及日誌
docker compose --profile vision --profile audio --profile synthetic ps
docker compose logs --tail=100 api
docker compose --profile vision logs --tail=100 vision-detector
docker compose --profile audio logs --tail=100 audio-detector

# 停止服務（保留資料與模型 volumes）
docker compose --profile vision --profile audio --profile synthetic down
```

- **模型不可用**：檢查 GPU、權重下載、暖機結果及對應服務日誌。服務的 health check 成功不一定代表模型已載入。
- **未偵測到人臉**：`no_face` 表示該影格沒有可分析的人臉，不代表媒體真實。
- **修改 `.env` 後未生效**：使用 `docker compose up -d --force-recreate api` 重新建立 API；涉及偵測服務的設定時也須重新建立對應服務。
- **QR Code 無法開啟**：確認 `PUBLIC_BASE_URL` 使用手機或其他裝置可連線的主機 IP，並重新建立 API。

## API

- `GET /health`：服務狀態
- `POST /api/analyze`：以 `multipart/form-data` 的 `file` 欄位上傳素材
- `GET /api/jobs/{job_id}`：查詢分析結果
- `GET /api/jobs/{job_id}/files/report.json`、`report.md`：下載報告
- `GET /r/{job_id}`、`GET /api/jobs/{job_id}/qr`：報告頁面及 QR Code

此專案是示範系統，報告網址沒有身分驗證或期限，請在受控網路使用，勿直接暴露至公網。正式部署前須補齊存取控制、限流、檔案安全與資料清理措施，並確認各上游模型與權重的授權條件。
