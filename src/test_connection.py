"""Test the SQL Server connection: create the database if missing, then write, read back and drop a test table."""
import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import pyodbc
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]  # project root
load_dotenv(ROOT / ".env")

SERVER = os.getenv("SQL_SERVER", ".")
DATABASE = os.getenv("SQL_DATABASE", "vn1_da")
DRIVER = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")


def conn_str(database: str) -> str:
    """Windows Authentication, no username or password needed."""
    return (
        f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={database};"
        "Trusted_Connection=yes;TrustServerCertificate=yes;"
    )


# 1. Create the database: run against master, CREATE DATABASE requires autocommit
create_sql = (ROOT / "sql" / "00_create_db.sql").read_text(encoding="utf-8")
conn = pyodbc.connect(conn_str("master"), autocommit=True)
conn.execute(create_sql, DATABASE)
conn.close()
print(f"Database {DATABASE} is ready")

# 2. Write, read back and drop a test table
engine = create_engine(
    "mssql+pyodbc:///?odbc_connect=" + quote_plus(conn_str(DATABASE)),
    fast_executemany=True,
)

test = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
test.to_sql("_connection_test", engine, if_exists="replace", index=False)

back = pd.read_sql("SELECT * FROM _connection_test", engine)
print(back)
assert len(back) == 3, "Unexpected row count when reading back"

with engine.begin() as c:
    c.execute(text("DROP TABLE _connection_test"))

print("Connection test passed: write, read and drop OK")