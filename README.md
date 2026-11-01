# 電商庫存與採購分析（VN1）

以美國線上賣家的真實週銷量資料，規劃 Q4 旺季的採購與定價。

> 🚧 進行中：目前完成環境建立與原始資料載入核對

## 資料

使用 VN1 Forecasting - Accuracy Challenge 資料集（46 個賣家、328 個倉庫、15,053 條商品序列、2020-07 ～ 2024-04 共 196 週）。

原始資料未開放授權，**不包含在本 repo**。請至 [DataSource.ai](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description) 註冊下載以下 5 個檔案，放進 `data/raw/`：

- `Phase 0 - Sales.csv`、`Phase 0 - Price.csv`
- `Phase 1 - Sales.csv`、`Phase 1 - Price.csv`
- `Phase 2 - Sales.csv`

## 環境需求

- Python 3.10 以上（開發時使用 3.14）
- SQL Server（Express 或 Developer 版，皆免費）
- ODBC Driver 18 for SQL Server

## 設定

```
pip install -r requirements.txt
copy .env.example .env
```

打開 `.env`，依自己的 SQL Server 修改連線設定。預設使用本機（`.`）與 Windows 驗證。

## 執行

```
python src/test_connection.py   # 建立資料庫 vn1_da 並測試讀寫
python src/load_raw.py          # 讀入 5 個檔案並核對資料
```

## 資料夾結構

```
├─ data/raw/     原始資料（不上傳）
├─ sql/          建表與查詢
├─ src/          Python 程式
└─ outputs/      產出（不上傳）
```

## 引用

Vandeput, Nicolas. "VN1 Forecasting - Accuracy Challenge." DataSource.ai, DataSource, 3 Oct. 2024, https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description
