# AI Truth Lab v0.3 — 整段 AI 生成影片檢測（DGX Spark / Docker）

此更新保留 v0.2 的 `api`、`ollama`、`vision-detector`、`audio-detector`，新增 **`synthetic-video-detector`**，使用 STALL + DINOv3 ViT-L/16 + VATEX 真實影片校準參數。原先 DeepGuard 繼續負責人臉修改，STALL 負責全畫面時空特徵是否異常；二者的分數不能相加。

**狀態**：Python 語法、Compose YAML、前端 JS 與 14 項本地程式測試通過；尚未在實際 DGX Spark GPU 上完成 STALL 建置、暖機、速度與準確度驗收。

## 0. 授權與資料安全

- STALL 官方程式使用 [CC BY-NC 4.0](https://github.com/OmerBenHayun/STALL/blob/main/LICENSE.txt)，僅允許符合其條件的非商業用途；記者會有商業推廣、廠商展示或贊助性質時，**務必先確認取得授權**。
- DINOv3 的程式與權重受 [Meta DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) 約束。權重由使用者自行向 Meta 取得，ZIP 中不提供模型權重。
- 這個專案現有 MVP 尚未加登入驗證：原始媒體、報告連結可能被知道 job ID 的人士存取。只放在受控內網，勿把 `8088` 直接公開至 Internet，也不要上傳未取得許可的人像或錄音。

## 1. 備份並更新既有程式（不更動原始 data / .env）

以下假設目前安裝目錄為 `~/Documents/ai-truth-lab`，而你已下載 `ai-truth-lab-v0.3-docker.zip` 到 `~/Documents`：

```bash
cd ~/Documents
cp -a ai-truth-lab ai-truth-lab-before-v03
unzip -o ai-truth-lab-v0.3-docker.zip -d v03-staging
# 若有自己改動過 app 程式，先用 diff 檢查後再覆蓋。
rsync -av --exclude='.env' --exclude='data/' --exclude='models/dinov3/*.pth' \
  v03-staging/ai-truth-lab/ ai-truth-lab/
cd ai-truth-lab
```

已有的 `.env` 需要**另外新增**一行（先保持關閉）：

```dotenv
SYNTHETIC_ENABLED=false
```

## 2. 官方 DINOv3 權重（手動取得）

進入 [Meta DINOv3 模型下載頁](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)，選擇 `ViT-L/16 distilled`，預訓練集 `LVD-1689M`，把授權取得的檔案放到：

```text
~/Documents/ai-truth-lab/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

檢查：

```bash
ls -lh models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

檔案不得是 HTML 錯誤頁、未完成下載或錯誤的模型變體。不要將權重加到 Git repository，勿未經授權分享。

## 3. 啟動獨立 STALL Docker 服務

```bash
sudo docker compose --profile synthetic build synthetic-video-detector
sudo docker compose --profile synthetic up -d synthetic-video-detector
sudo docker compose --profile synthetic ps
```

查無權重時 `/health` 仍可回 200（只是 service liveness）。實際請使用 `/warmup` 載入模型並確認 GPU：

```bash
sudo docker compose --profile synthetic exec -T synthetic-video-detector \
  python -c "import urllib.request; req=urllib.request.Request('http://127.0.0.1:8093/warmup',data=b'{}',method='POST'); print(urllib.request.urlopen(req,timeout=600).read().decode())"
```

預期 JSON 包含 `status:ready`、`device:cuda:0`。若是 503，先檢查權重路徑、授權檔案、DINOv3 依賴和容器 GPU 狀態：

```bash
sudo docker compose --profile synthetic logs --tail=120 synthetic-video-detector
sudo docker compose --profile synthetic exec -T synthetic-video-detector nvidia-smi
```

NGC 基底映像與 PyTorch/DINOv3 的相容性**仍需以實際 build/warmup 為準**。

## 4. 開啟 API 並保留原本兩種偵測器

```bash
# .env 設為 SYNTHETIC_ENABLED=true (保留 VISION_ENABLED=true / AUDIO_ENABLED=true)
sed -i 's/^SYNTHETIC_ENABLED=.*/SYNTHETIC_ENABLED=true/' .env
# 若從未加入過這個變數，請在 .env 新增一行 SYNTHETIC_ENABLED=true
sudo docker compose build api
sudo docker compose up -d --no-deps --force-recreate api
curl -s http://127.0.0.1:8088/health
```

預期 `version: "0.3.0"` 且 `synthetic_enabled: true`。**不需重建 DeepGuard / AASIST**。

## 5. 功能驗收

先用**自己有權使用的** 5–15 秒 MP4 影片測試（大於 20 秒會被第一階段的 API 拒絕）：

```bash
python3 scripts/smoke_synthetic.py /path/to/your_clip.mp4 http://127.0.0.1:8088
```

通過標準：`report.detectors.synthetic_video.status == complete` 且 `scored_windows > 0`。結果包含：

- `windows[].start_seconds / end_seconds` — 2 秒取樣視窗（不是竄改位置精確標記）
- `windows[].spatial_real_percentile` — 空間分支的真實影片參考百分位
- `windows[].temporal_real_percentile` — 時間分支的真實影片參考百分位
- `windows[].real_reference_score` — STALL 官方 final score，**越高越接近真實影片校準分布**
- `windows[].synthetic_anomaly_signal` = `1 - real_reference_score`，**不是影片由 AI 生成的機率**
- `windows[].review_flag` — 使用 `STALL_WARNING_THRESHOLD` 暫定門檻是否需要人工複核

根據 STALL 官方的參考演算法，對原始影片以 8 fps 取 2 秒視窗（16 影格），最多取片首／中／尾三個視窗以減少記者會等待時間。**多視窗分段是本專案增加的驗收方法，非 STALL 論文所驗證的精確定位演算法。**

介面會在既有 DeepGuard 時間軸下方顯示另一個 STALL 視窗列表，並於 QR Code 報告各自呈現。

## 6. 第二輪準確度驗收（記者會前不可略過）

至少準備：真實手機影片、全生成影片（來源明確，含不同生成器）、重壓縮影片、傳統人臉修改影片。分類必須有獨立真值標籤，記錄生成器、分辨率、壓縮方式。用相同權重及門檻跑盲測，報告 TPR/FPR、每種類型誤判和 DGX Spark 延遲。不能用兩支明顯換臉舊影片推導普遍準確率。

## 7. 關閉與回滾

只關 STALL，不影響 v0.2：

```bash
sudo docker compose --profile synthetic stop synthetic-video-detector
sed -i 's/^SYNTHETIC_ENABLED=.*/SYNTHETIC_ENABLED=false/' .env
sudo docker compose up -d --no-deps --force-recreate api
```

恢復原有 v0.2 時應備份當前 data 再使用 `ai-truth-lab-before-v03` 中的程式與 compose。**不要使用 `docker compose down -v`**，它可能刪除 Ollama 模型與工作資料。

## 來源

- STALL 官方：https://github.com/OmerBenHayun/STALL （Docker 透過 commit `f6dbb617a08d522f42ac02341028da8c58e669d1` 建置）
- STALL 校準參數：`precomputed/stall_params_vatex_dino_v3.npz`，隨官方 repository 下載
- DINOv3 官方：https://github.com/facebookresearch/dinov3
- Meta 授權權重：https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
