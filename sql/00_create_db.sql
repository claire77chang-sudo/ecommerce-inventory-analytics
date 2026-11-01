-- Create the project database (skipped if it already exists, safe to rerun)
-- The name is passed in by Python from SQL_DATABASE in .env
DECLARE @db sysname = ?;
DECLARE @sql nvarchar(max) = N'CREATE DATABASE ' + QUOTENAME(@db);
IF DB_ID(@db) IS NULL
    EXEC (@sql);
