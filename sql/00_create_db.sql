-- 建立專案資料庫（已存在就略過，可重複執行）
IF DB_ID('vn1_da') IS NULL
    CREATE DATABASE vn1_da;
