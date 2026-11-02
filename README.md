# 客服 Agent 系統 (RAG + Function Calling)

本專案實作一個支援 **RAG (檢索增強生成)** 與 **OpenAI Function Calling (多工具調用)** 的智慧客服 Agent。系統整合結構化資料庫（房型價格、店家資訊）與非結構化規章條款（`shop_rules.txt`），提供準確且具權威出處的顧客諮詢服務。

---

## 🚀 系統核心功能與架構

1. **結構化資料查詢 (SQL / Tools)**：
   * `get_shop_info`: 查詢營業時間、外送、優惠方案等基礎資訊。
   * `get_price`: 查詢特定房型或餐點品項的價格與空房/供應狀態。
2. **非結構化語意檢索 (RAG)**：
   * `search_shop_rules`: 將 10 項中英文雙語規章進行向量化（Embedding），透過 Cosine Similarity 進行語意檢索（門檻設定 `MIN_SCORE = 0.45`），並強制回答附帶來源標題。
3. **安全轉接機制**：
   * `transfer_to_human`: 當遭遇客訴、極端問題或規章未涵蓋（`found=False`）時自動切換專人服務。
4. **多工具協同 (Multi-Tool Calling)**：
   * 支援單一訊息中解析多個意圖，並平行調用不同工具（例如同時查詢「寵物規定」與「觀海雙人木屋價格」）。

---

## 📁 檔案結構

```text
├── shop_rules.txt         # 10 項中英雙語店內規章 (Markdown Chunk 格式)
├── room.csv               # 房型與價格資料庫
├── room_info.csv          # 房型詳細資訊
├── app.py                 # Agent 主程式邏輯與 API 整合
├── test_tool_choice.py    # Step 9 工具選擇自動化評測腳本
├── requirements.txt       # 專案依賴套件清單
├── .env                   # 金鑰與模型設定檔 (需自行建立)
└── README.md              # 專案說明文件
```

---

## 🛠️ 環境設定與安裝步驟

### 1. 切換 Conda 虛擬環境
```bash
conda activate ai26
```

### 2. 移動至專案目錄
```cmd
cd "C:\Users\linda\Desktop\深度學習\第8周"
```

### 3. 安裝依賴套件
```bash
pip install -r requirements.txt
```

### 4. 設定 `.env` 環境變數
在專案根目錄建立 `.env` 檔案，填入以下設定：
```env
API_KEY=your_openai_api_key_here
LLM_API_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini

EMBEDDING_API_KEY=your_openai_api_key_here
EMBEDDING_API_URL=https://api.openai.com/v1
EMBEDDING_MODEL_NAME=text-embedding-3-small
```

---

## 🧪 執行測試與評測

### 執行 Tool Choice 評測 (Step 9)
```bash
python test_tool_choice.py
```

預期評測結果：
* 工具選擇涵蓋 `get_shop_info`、`get_price`、`search_shop_rules` 與 `transfer_to_human`。
* 包含了 Rate Limit (429 Error) 的自動冷卻重試機制，確保評測穩定通過。