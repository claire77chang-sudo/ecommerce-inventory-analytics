# E-commerce Inventory & Purchasing Analytics (VN1)

Planning Q4 peak-season purchasing and pricing with real weekly sales data from US online sellers.

> 🚧 Work in progress: environment setup and raw data loading/validation are complete.

## Data

This project uses the VN1 Forecasting - Accuracy Challenge dataset (46 sellers, 328 warehouses, 15,053 product series, 196 weeks from July 2020 to April 2024).

The raw data has no open license and is **not included in this repo**. Register at [DataSource.ai](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description), download the following 5 files, and place them in `data/raw/`:

- `Phase 0 - Sales.csv`, `Phase 0 - Price.csv`
- `Phase 1 - Sales.csv`, `Phase 1 - Price.csv`
- `Phase 2 - Sales.csv`

## Requirements

- Python 3.10+ (developed on 3.14)
- SQL Server (Express or Developer edition, both free)
- ODBC Driver 18 for SQL Server

## Setup

```
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` to match your SQL Server:

| Variable | Description | Default |
| --- | --- | --- |
| `SQL_SERVER` | Server name (`.` = local machine) | `.` |
| `SQL_DATABASE` | Database name, created automatically if missing | `vn1_da` |
| `SQL_DRIVER` | ODBC driver name | `ODBC Driver 18 for SQL Server` |

Authentication uses Windows Authentication. To list installed ODBC drivers:

```
python -c "import pyodbc; print(pyodbc.drivers())"
```

## Run

```
python src/test_connection.py   # create database vn1_da and test read/write
python src/load_raw.py          # load the 5 files and validate the data
```

## Project Structure

```
├─ data/raw/     raw data (not uploaded)
├─ sql/          table creation and queries
├─ src/          Python scripts
└─ outputs/      outputs (not uploaded)
```

## Citation

Vandeput, Nicolas. "VN1 Forecasting - Accuracy Challenge." DataSource.ai, DataSource, 3 Oct. 2024, https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description
