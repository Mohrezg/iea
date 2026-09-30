# IEA Suite — Industrial Engineering Assistance

A full-stack Flask web application that bundles four industrial-engineering optimization tools — **Demand Forecasting**, **Inventory / Stock Management**, **Vehicle Routing (VRP)**, and **Flexible Job-Shop Production Scheduling (FJSP)** — behind a single shared SQLite database and a unified dashboard.

---

## ✨ Features

### 📊 Demand Forecast (CatBoost)
- Multi-product demand forecasting powered by **CatBoost** gradient-boosted trees.
- Automatic feature engineering: lags (1/2/3/7 days), rolling means/stds, calendar features (day-of-week, month, week-of-year), stock-to-demand ratio, plus optional `promotion`, `discount_percent` and `is_holiday` toggles.
- Historical data via CSV upload, a manual spreadsheet grid, **or directly from the shared database** (`/api/forecast/data`).
- Outputs metrics (MAE, RMSE, R², MAPE), test-vs-actual Plotly charts, future forecast tables and a ranked feature-importance visualization.
- Products are managed as tabs with per-product forecast filtering.

### 📦 Stock Management (ABC / XYZ / SS / RoP / EOQ)
- Complete inventory classification engine:
  - **ABC analysis** by annual consumption value (configurable thresholds).
  - **XYZ analysis** by demand variability / Coefficient of Variation.
  - Combined **ABC-XYZ** matrix.
  - **Safety Stock**, **Reorder Point** and **EOQ** per product, with selectable service levels (80 % → 99.9 %) or a custom Z-value.
- Demand-frequency auto-detection (daily / weekly / monthly / yearly) from CSV `Date` columns.
- Products & demand history are loaded from the database (editable inline per product, with charts), and results can be exported to CSV.
- Server-side validation reports row-level problems without blocking valid products.

### 🚚 Vehicle Routing (VRP)
- Heterogeneous fleet: each truck has its own **cost per km** and **per-product capacities**.
- Customer orders are aggregated per client; only ordering clients appear in the (asymmetric) **distance matrix**, editable inline and persisted to the DB.
- Two meta-heuristics selectable from the UI:
  - **Genetic Algorithm** (population size, generations, crossover / mutation rates).
  - **Variable Neighbourhood Search** (iterations, k_max).
- Interactive route map, per-route detail cards, and a customer-assignment table.

### 🏭 Production Scheduling (FJSP)
- **Flexible Job-Shop Scheduling** solved by an enhanced Genetic Algorithm:
  - SPT-heuristic seeded population, crowding tournament selection, adaptive mutation, cataclysmic mutation, periodic Tabu-search local improvement, diversity injection.
  - Chromosome encoding: machine assignment vector + random-key operation sequencing.
  - Termination by generation count **or by a time limit in seconds**.
- Workstation & machine management synced with the home dashboard; job routing built with an ordered click-to-select workstation strip.
- Results: Gantt chart (per machine), workstation load-contribution ranking, and a detailed operation table.

### 🏠 Home Dashboard
- **Operations Calendar** — pay days, restock days, machine-maintenance roll-forwards (workstation machines + standalone machines), client orders/deliveries, supplier bills, manual notes.
- **Stock / Products** — CRUD with images, SKU, qty, price, restock day, inventory cost parameters (unit cost, lead time, ordering cost, holding rate).
- **Clients** — CRUD, contact info, order history with statuses.
- **Employees** — CRUD, birthday-derived auto-age, hourly/daily pay, shift log, salary calculator, next-pay-date display.
- **Machines & Trucks** — workstations with per-machine maintenance metadata (last maintenance, interval), standalone machines, and trucks with capacities & cost/km.
- **Suppliers** — CRUD linked to provided products.
- **Bills of Orders** — line items, tax, statuses, and print-ready bill layout.

All entities are persisted in a shared **SQLite** database (`iea.db`) so every module sees the same clients, products, trucks, workstations and distances.

---

## 🗂️ Project Structure

```
project/
├── app.py                  # Flask application — all routes & API endpoints
│
├── database/
│   ├── __init__.py
│   └── db.py               # Shared SQLite layer (schema, CRUD, migrations)
│
├── solvers/
│   ├── classifier.py       # ABC/XYZ/SS/RoP/EOQ calculation engine
│   ├── CatBoost.py         # DemandForecaster (CatBoost regressor)
│   ├── fjsp.py             # Flexible Job-Shop GA + Tabu solver
│   └── vrp.py              # GA & VNS VRP heuristics (product capacities)
│
└── frontend/
    ├── home.html           # Dashboard (calendar, stock, clients, …)
    ├── forecast.html       # Demand forecast page
    ├── stock.html          # Inventory classification page
    ├── vrp.html            # Vehicle routing page
    └── fjsp.html           # Production scheduling page
```

---

## 🚀 Getting Started

### 1. Requirements

- Python **3.10+**
- pip packages:

```bash
pip install flask pandas numpy catboost scikit-learn openpyxl
```

> `deap` and `matplotlib` are used by the standalone solver scripts (`fjsp.py`).

### 2. Run

```bash
python app.py
```

The app starts on **http://0.0.0.0:5000** and creates `database/iea.db` automatically on first launch.

### 3. Use

| Page | URL |
|---|---|
| Home Dashboard | `http://localhost:5000/` |
| Demand Forecast | `http://localhost:5000/forecast` |
| Stock Management | `http://localhost:5000/stock` |
| Vehicle Routing | `http://localhost:5000/vrp` |
| Production Scheduling | `http://localhost:5000/scheduling` |

**Typical workflow:**

1. Populate **Products / Clients / Trucks / Workstations** on the Home dashboard.
2. Add **demand history** to products (Stock page CSV import: `Product_ID, Product_Name, Date, Demand, Unit_Cost, Lead_Time_Days, Ordering_Cost, Holding_Cost_Rate`).
3. Run the **Forecast** page (data can be pulled straight from the DB).
4. Run **Stock** calculations → ABC/XYZ classes, safety stock, reorder points, EOQ.
5. Create **customer orders** and fill the **distance matrix** on the VRP page → solve with GA or VNS.
6. Define **workstations, routings and jobs** on the Scheduling page → solve the FJSP and view the Gantt chart.

---

## 🔌 API Overview

| Endpoint | Method | Description |
|---|---|---|
| `/api/schedule` | POST | Solve FJSP (workstations, routings, processing times, time limit) |
| `/api/vrp/solve` | POST | Solve VRP (customers, vehicles, distance matrix, algorithm, params) |
| `/api/calculate` | POST | ABC/XYZ/SS/RoP/EOQ pipeline (JSON products or CSV upload) |
| `/api/detect-frequency` | POST | Auto-detect demand frequency from CSV dates |
| `/api/forecast` | POST | One-shot train + forecast (payload from forecast.html) |
| `/api/forecast/train` · `/predict` · `/status` · `/feature_importance` · `/event_impact` | — | Granular CatBoost endpoints |
| `/api/forecast/data` | GET | Flat demand-history dump from the DB (`?product_id=` filter) |
| `/api/products` (+ `/with-demand`, `/<id>/demand`, `/import-csv`) | GET/POST/PUT/DELETE | Product CRUD + demand history |
| `/api/clients` | GET/POST/PUT/DELETE | Client CRUD |
| `/api/trucks` | GET/POST/PUT/DELETE | Truck CRUD |
| `/api/suppliers` | GET/POST/PUT/DELETE | Supplier CRUD (linked to products) |
| `/api/employees` | GET/POST/PUT/DELETE | Employee CRUD + shift log reconciliation |
| `/api/workstations` (+ `/create`, `/<id>`) | GET/POST/PUT/DELETE | Workstation & machine CRUD (used by FJSP + calendar) |
| `/api/distances` | GET/POST | Asymmetric client↔client distance matrix (0 = depot) |

---

## 🧮 Solver Notes

- **FJSP** (`solvers/fjsp.py`): DEAP-based GA. Runs either for `ngen` generations or until `time_limit` seconds elapse; Hall-of-Fame tracking with adaptive mutation on stagnation and low diversity.
- **VRP** (`solvers/vrp.py`): permutation encoding evaluated against multi-product vehicle capacities; cheapest-first vehicle assignment; GA (tournament selection + OX crossover) or VNS (shake + swap local search).
- **Classifier** (`solvers/classifier.py`): pure NumPy/Pandas statistics; sample std-dev (ddof=1) for CV and safety stock, `Z · σd · √LT` safety stock, classic EOQ `√(2DS/H)`.
- **CatBoost** (`solvers/CatBoost.py`): group-aware lag/rolling features (no look-ahead), chronological train/test split, `product_id` treated as a categorical feature.

---

## ⚙️ Configuration

- **Database path**: set the `IEA_DB_PATH` environment variable to override the default `database/iea.db`.
- The SQLite schema self-creates on startup and auto-migrates older databases (e.g. adds missing `products` columns, rebuilds `client_distances` without foreign keys so the depot sentinel `0` works).
- Employee ages are refreshed from birthdays on every startup.

---

## 📄 contact me at

- **Email@**: rezguimohamedabdallah@gmail.com

