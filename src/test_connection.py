
"""測試 SQL Server 連線：建資料庫（不存在才建）→ 寫入測試表 → 讀回 → 刪除"""
import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import pyodbc
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄
load_dotenv(ROOT / ".env")

SERVER = os.getenv("SQL_SERVER", ".")
DATABASE = os.getenv("SQL_DATABASE", "vn1_da")
DRIVER = os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server")


def conn_str(database: str) -> str:
    """Windows 驗證登入，不需帳號密碼"""
    return (
        f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={database};"
        "Trusted_Connection=yes;TrustServerCertificate=yes;"
    )


# 1. 建資料庫：連到系統資料庫 master 執行，CREATE DATABASE 必須 autocommit
create_sql = (ROOT / "sql" / "00_create_db.sql").read_text(encoding="utf-8")
conn = pyodbc.connect(conn_str("master"), autocommit=True)
conn.execute(create_sql)
conn.close()
print(f"資料庫 {DATABASE} 已就緒")

# 2. 寫入、讀回、刪除測試表
engine = create_engine(
    "mssql+pyodbc:///?odbc_connect=" + quote_plus(conn_str(DATABASE)),
    fast_executemany=True,
)

test = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
test.to_sql("_connection_test", engine, if_exists="replace", index=False)

back = pd.read_sql("SELECT * FROM _connection_test", engine)
print(back)
assert len(back) == 3, "讀回筆數不對"

with engine.begin() as c:
    c.execute(text("DROP TABLE _connection_test"))

print("連線測試成功：可寫入、讀取、刪除")