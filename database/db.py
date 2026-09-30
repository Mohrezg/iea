"""
IEA Suite — shared SQLite data layer.

Stores workstations + their individual machines (with maintenance metadata)
so that the FJSP page and the home dashboard stay in sync.

Public API
----------
    init_db()
    # Workstations
    get_workstations()                       -> list[dict]  (with machine count)
    get_workstation(ws_id)                   -> dict | None (with machines list)
    replace_workstations(items)              -> list[dict]  (bulk, from FJSP)
    create_workstation(name, machines_count) -> dict
    update_workstation(ws_id, name, machines)-> dict | None
    delete_workstation(ws_id)                -> None
    clear_workstations()                     -> None

    # Products
    get_products() / get_product(pid)
    create_product(...) / update_product(...) / delete_product(pid)

    # Product demand history  (NEW)
    get_product_demand(pid)
    save_product_demand(pid, records, replace=True)
    get_products_with_demand()
    upsert_product_with_demand(product_id, name, ...)

    # Clients
    get_clients() / get_client(cid)
    create_client(...) / update_client(...) / delete_client(cid)

    # Trucks
    get_trucks() / get_truck(tid)
    create_truck(...) / update_truck(...) / delete_truck(tid)

    # Suppliers
    get_suppliers() / get_supplier(sid)
    create_supplier(...) / update_supplier(...) / delete_supplier(sid)

    # Employees
    get_employees() / get_employee(eid)
    create_employee(...) / update_employee(...) / delete_employee(eid)
    refresh_all_ages()                       -> int  (recompute ages from birthday)

    # Distances
    get_distance_matrix() / set_distance(...) / set_distance_matrix(...)
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from typing import Iterable

_DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("IEA_DB_PATH", os.path.join(_DB_DIR, "iea.db"))


# ───────────────────────────────────────────────────────────────────────
#  Connection plumbing
# ───────────────────────────────────────────────────────────────────────
@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ───────────────────────────────────────────────────────────────────────
#  Schema
# ───────────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS workstations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    position    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_workstations_position ON workstations(position);

CREATE TABLE IF NOT EXISTS machines (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    workstation_id    INTEGER NOT NULL,
    name              TEXT    NOT NULL,
    last_maintenance  TEXT,
    interval_days     INTEGER NOT NULL DEFAULT 30,
    notes             TEXT    NOT NULL DEFAULT '',
    position          INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (workstation_id) REFERENCES workstations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_machines_workstation
    ON machines(workstation_id, position);

CREATE TABLE IF NOT EXISTS products (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT    NOT NULL,
    sku               TEXT    NOT NULL DEFAULT '',
    qty               INTEGER NOT NULL DEFAULT 0,
    unit              TEXT    NOT NULL DEFAULT 'pcs',
    price             REAL    NOT NULL DEFAULT 0,
    unit_cost         REAL    NOT NULL DEFAULT 0,
    lead_time_days    REAL    NOT NULL DEFAULT 0,
    ordering_cost     REAL    NOT NULL DEFAULT 0,
    holding_cost_rate REAL    NOT NULL DEFAULT 0.25,
    restock_day       INTEGER,
    image             TEXT,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_products_name ON products(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_products_sku  ON products(sku  COLLATE NOCASE);

-- Historical demand per product (used by the Stock page)
CREATE TABLE IF NOT EXISTS product_demand_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id  INTEGER NOT NULL,
    date        TEXT,
    demand      REAL NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_product_demand_product
    ON product_demand_history(product_id, date);

CREATE TABLE IF NOT EXISTS clients (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    contact     TEXT    NOT NULL DEFAULT '',
    email       TEXT    NOT NULL DEFAULT '',
    phone       TEXT    NOT NULL DEFAULT '',
    address     TEXT    NOT NULL DEFAULT '',
    image       TEXT,
    position    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_clients_position ON clients(position);

CREATE TABLE IF NOT EXISTS trucks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    cost_per_km REAL    NOT NULL DEFAULT 1.0,
    capacities  TEXT    NOT NULL DEFAULT '{}',
    notes       TEXT    NOT NULL DEFAULT '',
    image       TEXT,
    position    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_trucks_position ON trucks(position);

-- NOTE: no FOREIGN KEYs here on purpose. from_client_id / to_client_id use
-- 0 as a sentinel for the depot, which is not a real client row.
CREATE TABLE IF NOT EXISTS client_distances (
    from_client_id INTEGER NOT NULL,
    to_client_id   INTEGER NOT NULL,
    distance_km    REAL,
    PRIMARY KEY (from_client_id, to_client_id)
);

-- ── Suppliers ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS suppliers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    contact     TEXT    NOT NULL DEFAULT '',
    email       TEXT    NOT NULL DEFAULT '',
    phone       TEXT    NOT NULL DEFAULT '',
    address     TEXT    NOT NULL DEFAULT '',
    image       TEXT,
    position    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_suppliers_position ON suppliers(position);

-- Many-to-many: which products a supplier provides.
CREATE TABLE IF NOT EXISTS supplier_products (
    supplier_id INTEGER NOT NULL,
    product_id  INTEGER NOT NULL,
    PRIMARY KEY (supplier_id, product_id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(id) ON DELETE CASCADE,
    FOREIGN KEY (product_id)  REFERENCES products(id)  ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_supplier_products_product
    ON supplier_products(product_id);

-- ── Employees ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS employees (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    age         INTEGER,
    sex         TEXT    NOT NULL DEFAULT '',
    birthday    TEXT,
    email       TEXT    NOT NULL DEFAULT '',
    phone       TEXT    NOT NULL DEFAULT '',
    address     TEXT    NOT NULL DEFAULT '',
    image       TEXT,
    pay_type    TEXT    NOT NULL DEFAULT 'hourly',
    rate        REAL    NOT NULL DEFAULT 0,
    pay_day     INTEGER,
    position    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_employees_position ON employees(position);

CREATE TABLE IF NOT EXISTS employee_shifts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL,
    date        TEXT    NOT NULL,
    hours       REAL    NOT NULL DEFAULT 1,
    FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_employee_shifts_employee
    ON employee_shifts(employee_id, date);
"""


def _migrate_client_distances() -> None:
    """
    Older versions of client_distances had FOREIGN KEY constraints referencing
    clients(id). But we use 0 as the depot sentinel, which isn't a client, so
    inserts violate the FK. Drop the old table once so _SCHEMA can rebuild it
    without FKs.
    """
    with _conn() as c:
        row = c.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='client_distances'"
        ).fetchone()
        if row and row["sql"] and "FOREIGN KEY" in row["sql"].upper():
            c.execute("DROP TABLE client_distances")


def _migrate_products_price() -> None:
    """Add products.price if an older DB exists without it."""
    with _conn() as c:
        cols = [r["name"] for r in c.execute("PRAGMA table_info(products)").fetchall()]
        if cols and "price" not in cols:
            c.execute("ALTER TABLE products ADD COLUMN price REAL NOT NULL DEFAULT 0")


def _migrate_products_inventory() -> None:
    """Add inventory-related columns to products if an older DB lacks them."""
    with _conn() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(products)").fetchall()}
        if not cols:
            return
        for col, ddl in [
            ("unit_cost",         "REAL NOT NULL DEFAULT 0"),
            ("lead_time_days",    "REAL NOT NULL DEFAULT 0"),
            ("ordering_cost",     "REAL NOT NULL DEFAULT 0"),
            ("holding_cost_rate", "REAL NOT NULL DEFAULT 0.25"),
        ]:
            if col not in cols:
                c.execute(f"ALTER TABLE products ADD COLUMN {col} {ddl}")


def init_db() -> None:
    """Create missing tables. Safe to call on every app start."""
    _migrate_client_distances()
    with _conn() as c:
        c.executescript(_SCHEMA)
    _migrate_products_price()
    _migrate_products_inventory()

    # keep employee ages in sync with today's date
    try:
        refresh_all_ages()
    except Exception:
        # Never let a data issue block startup
        pass


# ───────────────────────────────────────────────────────────────────────
#  Workstations
# ───────────────────────────────────────────────────────────────────────
def _ws_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id":       row["id"],
        "name":     row["name"],
        "position": row["position"],
        "machines": row["machines"],
    }


def get_workstations() -> list[dict]:
    """All workstations with their machine count, ordered by position."""
    with _conn() as c:
        rows = c.execute("""
            SELECT w.id, w.name, w.position,
                   (SELECT COUNT(*) FROM machines m
                     WHERE m.workstation_id = w.id) AS machines
            FROM workstations w
            ORDER BY w.position, w.id
        """).fetchall()
        return [_ws_row_to_dict(r) for r in rows]


def get_workstation(ws_id: int) -> dict | None:
    """A single workstation with its full machines list."""
    with _conn() as c:
        ws = c.execute(
            "SELECT id, name, position FROM workstations WHERE id = ?",
            (ws_id,)
        ).fetchone()
        if not ws:
            return None
        machines = c.execute("""
            SELECT id, name, last_maintenance, interval_days, notes, position
            FROM machines
            WHERE workstation_id = ?
            ORDER BY position, id
        """, (ws_id,)).fetchall()
        return {
            "id":       ws["id"],
            "name":     ws["name"],
            "position": ws["position"],
            "machines": [dict(m) for m in machines],
        }


def _default_machine_name(ws_name: str, idx: int) -> str:
    return f"{ws_name}-M{idx + 1}"


def replace_workstations(items: Iterable[dict]) -> list[dict]:
    """
    Replace the entire workstation set (bulk push from FJSP page).

    Machine rows are preserved by POSITION, so renaming a workstation keeps
    its maintenance metadata intact.
    """
    cleaned: list[tuple[str, int, int]] = []
    for i, item in enumerate(items or []):
        name = str((item or {}).get("name", "")).strip()
        if not name:
            continue
        try:
            mcount = int((item or {}).get("machines", 1))
        except (TypeError, ValueError):
            mcount = 1
        cleaned.append((name, max(1, mcount), i))

    with _conn() as c:
        # Snapshot machines by (ws_position, machine_position)
        snapshot: dict[tuple[int, int], dict] = {}
        for row in c.execute("""
            SELECT w.position AS ws_pos, m.position AS m_pos,
                   m.name AS m_name, m.last_maintenance,
                   m.interval_days, m.notes
            FROM machines m JOIN workstations w ON w.id = m.workstation_id
        """).fetchall():
            snapshot[(row["ws_pos"], row["m_pos"])] = dict(row)

        c.execute("DELETE FROM machines")
        c.execute("DELETE FROM workstations")

        for name, mcount, pos in cleaned:
            cur = c.execute(
                "INSERT INTO workstations (name, position) VALUES (?, ?)",
                (name, pos),
            )
            ws_id = cur.lastrowid
            for m_idx in range(mcount):
                old = snapshot.get((pos, m_idx))
                if old:
                    c.execute("""
                        INSERT INTO machines
                            (workstation_id, name, last_maintenance,
                             interval_days, notes, position)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (ws_id, old["m_name"], old["last_maintenance"],
                          old["interval_days"], old["notes"], m_idx))
                else:
                    c.execute("""
                        INSERT INTO machines
                            (workstation_id, name, interval_days, notes, position)
                        VALUES (?, ?, 30, '', ?)
                    """, (ws_id, _default_machine_name(name, m_idx), m_idx))

    return get_workstations()


def create_workstation(name: str, machines_count: int = 1) -> dict:
    """Append a new workstation with N default machines."""
    name = str(name or "").strip()
    if not name:
        raise ValueError("Workstation name is required.")
    try:
        mcount = max(1, int(machines_count))
    except (TypeError, ValueError):
        mcount = 1

    with _conn() as c:
        max_pos = c.execute(
            "SELECT COALESCE(MAX(position), -1) AS p FROM workstations"
        ).fetchone()["p"]
        cur = c.execute(
            "INSERT INTO workstations (name, position) VALUES (?, ?)",
            (name, max_pos + 1),
        )
        ws_id = cur.lastrowid
        for i in range(mcount):
            c.execute("""
                INSERT INTO machines
                    (workstation_id, name, interval_days, notes, position)
                VALUES (?, ?, 30, '', ?)
            """, (ws_id, _default_machine_name(name, i), i))

    return get_workstation(ws_id)


def update_workstation(ws_id: int, name: str, machines: list[dict]) -> dict | None:
    """
    Update a workstation's name and reconcile its machine list.

    `machines` is a list of dicts:
        {"id": 1, "name": "Cutting-A", "last_maintenance": "2026-09-01",
         "interval_days": 30, "notes": ""}
    A machine with no `id` gets created; machines absent from the list get
    deleted.
    """
    with _conn() as c:
        ws = c.execute("SELECT id, name FROM workstations WHERE id = ?",
                       (ws_id,)).fetchone()
        if not ws:
            return None

        new_name = str(name or "").strip() or ws["name"]
        c.execute(
            "UPDATE workstations SET name = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (new_name, ws_id),
        )

        existing_ids = {row["id"] for row in c.execute(
            "SELECT id FROM machines WHERE workstation_id = ?", (ws_id,)
        ).fetchall()}

        kept_ids: set[int] = set()
        for i, m in enumerate(machines or []):
            mid = m.get("id")
            mname = str(m.get("name") or "").strip() or _default_machine_name(new_name, i)
            lm = m.get("last_maintenance") or None
            try:
                interval = max(1, int(m.get("interval_days") or 30))
            except (TypeError, ValueError):
                interval = 30
            notes = str(m.get("notes") or "")

            if mid and mid in existing_ids:
                c.execute("""
                    UPDATE machines
                       SET name = ?, last_maintenance = ?, interval_days = ?,
                           notes = ?, position = ?, updated_at = datetime('now')
                     WHERE id = ? AND workstation_id = ?
                """, (mname, lm, interval, notes, i, mid, ws_id))
                kept_ids.add(mid)
            else:
                cur = c.execute("""
                    INSERT INTO machines
                        (workstation_id, name, last_maintenance,
                         interval_days, notes, position)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (ws_id, mname, lm, interval, notes, i))
                kept_ids.add(cur.lastrowid)

        # Drop machines the user removed
        for mid in (existing_ids - kept_ids):
            c.execute("DELETE FROM machines WHERE id = ?", (mid,))

    return get_workstation(ws_id)


def delete_workstation(ws_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM machines WHERE workstation_id = ?", (ws_id,))
        c.execute("DELETE FROM workstations WHERE id = ?", (ws_id,))


def clear_workstations() -> None:
    with _conn() as c:
        c.execute("DELETE FROM machines")
        c.execute("DELETE FROM workstations")


# ───────────────────────────────────────────────────────────────────────
#  Products (Stock tab <-> FJSP job names)
# ───────────────────────────────────────────────────────────────────────
def _product_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id":                row["id"],
        "name":              row["name"],
        "sku":               row["sku"] or "",
        "qty":               row["qty"] or 0,
        "unit":              row["unit"] or "pcs",
        "price":             float(row["price"] or 0),
        "unit_cost":         float(row["unit_cost"] or 0),
        "lead_time_days":    float(row["lead_time_days"] or 0),
        "ordering_cost":     float(row["ordering_cost"] or 0),
        "holding_cost_rate": float(row["holding_cost_rate"] or 0.25),
        "restock_day":       row["restock_day"],
        "image":             row["image"],
    }


def get_products() -> list[dict]:
    with _conn() as c:
        rows = c.execute("""
            SELECT id, name, sku, qty, unit, price,
                   unit_cost, lead_time_days, ordering_cost, holding_cost_rate,
                   restock_day, image
            FROM products
            ORDER BY name COLLATE NOCASE, id
        """).fetchall()
        return [_product_row_to_dict(r) for r in rows]


def get_product(pid: int) -> dict | None:
    with _conn() as c:
        row = c.execute("""
            SELECT id, name, sku, qty, unit, price,
                   unit_cost, lead_time_days, ordering_cost, holding_cost_rate,
                   restock_day, image
            FROM products WHERE id = ?
        """, (pid,)).fetchone()
        return _product_row_to_dict(row) if row else None


def _normalize_restock_day(v):
    try:
        d = int(v)
    except (TypeError, ValueError):
        return None
    return d if 1 <= d <= 31 else None


def _normalize_price(v) -> float:
    try:
        p = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, p)


def _normalize_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def create_product(name, sku="", qty=0, unit="pcs",
                   price=0.0, restock_day=None, image=None,
                   unit_cost=0.0, lead_time_days=0.0,
                   ordering_cost=0.0, holding_cost_rate=0.25) -> dict:
    name = str(name or "").strip()
    if not name:
        raise ValueError("Product name is required.")
    try:
        qty = max(0, int(qty or 0))
    except (TypeError, ValueError):
        qty = 0
    with _conn() as c:
        cur = c.execute("""
            INSERT INTO products
                (name, sku, qty, unit, price,
                 unit_cost, lead_time_days, ordering_cost, holding_cost_rate,
                 restock_day, image)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, str(sku or "").strip(), qty,
              str(unit or "pcs").strip() or "pcs",
              _normalize_price(price),
              max(0.0, _normalize_float(unit_cost)),
              max(0.0, _normalize_float(lead_time_days)),
              max(0.0, _normalize_float(ordering_cost)),
              max(0.0, _normalize_float(holding_cost_rate, 0.25)),
              _normalize_restock_day(restock_day), image))
        pid = cur.lastrowid
    return get_product(pid)


def update_product(pid: int, name, sku="", qty=0, unit="pcs",
                   price=0.0, restock_day=None, image=None,
                   unit_cost=0.0, lead_time_days=0.0,
                   ordering_cost=0.0, holding_cost_rate=0.25) -> dict | None:
    name = str(name or "").strip()
    if not name:
        raise ValueError("Product name is required.")
    try:
        qty = max(0, int(qty or 0))
    except (TypeError, ValueError):
        qty = 0
    with _conn() as c:
        row = c.execute("SELECT id FROM products WHERE id = ?", (pid,)).fetchone()
        if not row:
            return None
        c.execute("""
            UPDATE products
               SET name = ?, sku = ?, qty = ?, unit = ?, price = ?,
                   unit_cost = ?, lead_time_days = ?,
                   ordering_cost = ?, holding_cost_rate = ?,
                   restock_day = ?, image = ?,
                   updated_at = datetime('now')
             WHERE id = ?
        """, (name, str(sku or "").strip(), qty,
              str(unit or "pcs").strip() or "pcs",
              _normalize_price(price),
              max(0.0, _normalize_float(unit_cost)),
              max(0.0, _normalize_float(lead_time_days)),
              max(0.0, _normalize_float(ordering_cost)),
              max(0.0, _normalize_float(holding_cost_rate, 0.25)),
              _normalize_restock_day(restock_day), image, pid))
    return get_product(pid)


def delete_product(pid: int) -> None:
    with _conn() as c:
        # supplier_products rows cascade via FK.
        c.execute("DELETE FROM products WHERE id = ?", (pid,))


# ───────────────────────────────────────────────────────────────────────
#  Product demand history
# ───────────────────────────────────────────────────────────────────────
def get_product_demand(pid: int) -> list[dict]:
    """Return [{'id', 'date', 'demand'}, ...] for a product, ordered by date."""
    with _conn() as c:
        rows = c.execute("""
            SELECT id, date, demand FROM product_demand_history
            WHERE product_id = ? ORDER BY date IS NULL, date, id
        """, (pid,)).fetchall()
        return [{"id": r["id"], "date": r["date"], "demand": float(r["demand"])}
                for r in rows]


def save_product_demand(pid: int, records, replace: bool = True) -> int:
    """
    Persist demand records for a product.
    records = [{'date': 'YYYY-MM-DD'|None, 'demand': float}, ...]
    Returns the number of rows written.
    """
    with _conn() as c:
        exists = c.execute("SELECT 1 FROM products WHERE id = ?", (pid,)).fetchone()
        if not exists:
            raise ValueError(f"Product {pid} not found.")
        if replace:
            c.execute("DELETE FROM product_demand_history WHERE product_id = ?", (pid,))
        n = 0
        for r in records or []:
            try:
                d = float(r.get("demand"))
            except (TypeError, ValueError):
                continue
            if d < 0:
                continue
            date_val = r.get("date") or None
            c.execute("""
                INSERT INTO product_demand_history (product_id, date, demand)
                VALUES (?, ?, ?)
            """, (pid, date_val, d))
            n += 1
        return n


def get_products_with_demand() -> list[dict]:
    """All products, each enriched with its full demand history."""
    products = get_products()
    if not products:
        return []
    with _conn() as c:
        rows = c.execute("""
            SELECT product_id, id, date, demand
            FROM product_demand_history
            ORDER BY product_id, date IS NULL, date, id
        """).fetchall()
    by_pid: dict[int, list[dict]] = {}
    for r in rows:
        by_pid.setdefault(r["product_id"], []).append({
            "id": r["id"], "date": r["date"], "demand": float(r["demand"]),
        })
    for p in products:
        p["demand_history"] = by_pid.get(p["id"], [])
    return products


def upsert_product_with_demand(
    product_id: str,
    name: str,
    unit_cost: float = 0.0,
    lead_time_days: float = 0.0,
    ordering_cost: float = 0.0,
    holding_cost_rate: float = 0.25,
    records=None,
) -> dict:
    """
    Create a product (or update an existing one matched by SKU == product_id)
    and replace its demand history.
    """
    product_id = str(product_id or "").strip()
    if not product_id:
        raise ValueError("product_id is required.")
    name = str(name or product_id).strip() or product_id

    with _conn() as c:
        row = c.execute(
            "SELECT id FROM products WHERE sku = ? COLLATE NOCASE",
            (product_id,),
        ).fetchone()
        if row:
            pid = row["id"]
            c.execute("""
                UPDATE products
                   SET name = ?, unit_cost = ?, lead_time_days = ?,
                       ordering_cost = ?, holding_cost_rate = ?,
                       updated_at = datetime('now')
                 WHERE id = ?
            """, (name,
                  max(0.0, _normalize_float(unit_cost)),
                  max(0.0, _normalize_float(lead_time_days)),
                  max(0.0, _normalize_float(ordering_cost)),
                  max(0.0, _normalize_float(holding_cost_rate, 0.25)),
                  pid))
        else:
            cur = c.execute("""
                INSERT INTO products
                    (name, sku, qty, unit, price,
                     unit_cost, lead_time_days, ordering_cost, holding_cost_rate)
                VALUES (?, ?, 0, 'pcs', 0, ?, ?, ?, ?)
            """, (name, product_id,
                  max(0.0, _normalize_float(unit_cost)),
                  max(0.0, _normalize_float(lead_time_days)),
                  max(0.0, _normalize_float(ordering_cost)),
                  max(0.0, _normalize_float(holding_cost_rate, 0.25))))
            pid = cur.lastrowid

    save_product_demand(pid, records or [], replace=True)
    p = get_product(pid)
    p["demand_history"] = get_product_demand(pid)
    return p


# ───────────────────────────────────────────────────────────────────────
#  Trucks
# ───────────────────────────────────────────────────────────────────────
def _truck_row_to_dict(row: sqlite3.Row) -> dict:
    try:
        caps = json.loads(row["capacities"] or "{}")
    except Exception:
        caps = {}
    return {
        "id":          row["id"],
        "name":        row["name"],
        "cost_per_km": float(row["cost_per_km"] or 0),
        "capacities":  caps,
        "notes":       row["notes"] or "",
        "image":       row["image"],
    }


def get_trucks() -> list[dict]:
    with _conn() as c:
        rows = c.execute("""
            SELECT id, name, cost_per_km, capacities, notes, image
            FROM trucks ORDER BY position, id
        """).fetchall()
        return [_truck_row_to_dict(r) for r in rows]


def get_truck(tid: int) -> dict | None:
    with _conn() as c:
        row = c.execute("""
            SELECT id, name, cost_per_km, capacities, notes, image
            FROM trucks WHERE id = ?
        """, (tid,)).fetchone()
        return _truck_row_to_dict(row) if row else None


def create_truck(name, cost_per_km=1.0, capacities=None, notes="", image=None) -> dict:
    name = str(name or "").strip()
    if not name:
        raise ValueError("Truck name is required.")
    try:
        cost = max(0.0, float(cost_per_km or 0))
    except (TypeError, ValueError):
        cost = 1.0
    caps_json = json.dumps(capacities or {})
    with _conn() as c:
        max_pos = c.execute(
            "SELECT COALESCE(MAX(position), -1) AS p FROM trucks"
        ).fetchone()["p"]
        cur = c.execute("""
            INSERT INTO trucks (name, cost_per_km, capacities, notes, image, position)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, cost, caps_json, notes or "", image, max_pos + 1))
        tid = cur.lastrowid
    return get_truck(tid)


def update_truck(tid: int, name, cost_per_km, capacities,
                 notes="", image=None) -> dict | None:
    name = str(name or "").strip()
    if not name:
        raise ValueError("Truck name is required.")
    try:
        cost = max(0.0, float(cost_per_km or 0))
    except (TypeError, ValueError):
        cost = 1.0
    caps_json = json.dumps(capacities or {})
    with _conn() as c:
        row = c.execute("SELECT id FROM trucks WHERE id = ?", (tid,)).fetchone()
        if not row:
            return None
        c.execute("""
            UPDATE trucks
               SET name = ?, cost_per_km = ?, capacities = ?, notes = ?,
                   image = ?, updated_at = datetime('now')
             WHERE id = ?
        """, (name, cost, caps_json, notes or "", image, tid))
    return get_truck(tid)


def delete_truck(tid: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM trucks WHERE id = ?", (tid,))


# ───────────────────────────────────────────────────────────────────────
#  Clients  (declared on the Home page, consumed by VRP)
# ───────────────────────────────────────────────────────────────────────
def _client_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id":      row["id"],
        "name":    row["name"],
        "contact": row["contact"] or "",
        "email":   row["email"] or "",
        "phone":   row["phone"] or "",
        "address": row["address"] or "",
        "image":   row["image"],
    }


def get_clients() -> list[dict]:
    with _conn() as c:
        rows = c.execute("""
            SELECT id, name, contact, email, phone, address, image
            FROM clients ORDER BY position, id
        """).fetchall()
        return [_client_row_to_dict(r) for r in rows]


def get_client(cid: int) -> dict | None:
    with _conn() as c:
        row = c.execute("""
            SELECT id, name, contact, email, phone, address, image
            FROM clients WHERE id = ?
        """, (cid,)).fetchone()
        return _client_row_to_dict(row) if row else None


def _seed_distance_rows(c: sqlite3.Connection, new_id: int) -> None:
    """
    Create NULL placeholders so a new client appears in the distance matrix.
    The depot is represented by client id 0 (a sentinel — not a real client).
    """
    others = [r["id"] for r in c.execute(
        "SELECT id FROM clients WHERE id != ?", (new_id,)
    ).fetchall()]
    pairs = set()
    for other in [0] + others:
        pairs.add((0, other))
        pairs.add((other, 0))
        pairs.add((new_id, other))
        pairs.add((other, new_id))
    for f, t in pairs:
        if f == t == 0:
            continue
        c.execute("""
            INSERT OR IGNORE INTO client_distances
                (from_client_id, to_client_id, distance_km)
            VALUES (?, ?, NULL)
        """, (f, t))


def create_client(name, contact="", email="", phone="",
                  address="", image=None) -> dict:
    name = str(name or "").strip()
    if not name:
        raise ValueError("Client name is required.")
    with _conn() as c:
        max_pos = c.execute(
            "SELECT COALESCE(MAX(position), -1) AS p FROM clients"
        ).fetchone()["p"]
        cur = c.execute("""
            INSERT INTO clients (name, contact, email, phone, address, image, position)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, contact or "", email or "", phone or "",
              address or "", image, max_pos + 1))
        cid = cur.lastrowid
        _seed_distance_rows(c, cid)
    return get_client(cid)


def update_client(cid, name, contact="", email="", phone="",
                  address="", image=None) -> dict | None:
    name = str(name or "").strip()
    if not name:
        raise ValueError("Client name is required.")
    with _conn() as c:
        row = c.execute("SELECT id FROM clients WHERE id = ?", (cid,)).fetchone()
        if not row:
            return None
        c.execute("""
            UPDATE clients
               SET name = ?, contact = ?, email = ?, phone = ?, address = ?,
                   image = ?, updated_at = datetime('now')
             WHERE id = ?
        """, (name, contact or "", email or "", phone or "",
              address or "", image, cid))
    return get_client(cid)


def delete_client(cid: int) -> None:
    with _conn() as c:
        # Clean up distance rows manually — the table has no FKs by design.
        c.execute(
            "DELETE FROM client_distances "
            "WHERE from_client_id = ? OR to_client_id = ?",
            (cid, cid),
        )
        c.execute("DELETE FROM clients WHERE id = ?", (cid,))


# ───────────────────────────────────────────────────────────────────────
#  Client ↔ client distances  (0 = depot)
# ───────────────────────────────────────────────────────────────────────
def get_distance_matrix() -> dict:
    """
    Return {"clients": [...], "ids": [...], "matrix": [[..]], "n": n}.

    matrix is (n+1)x(n+1); index 0 is the depot, indices 1..n follow client
    order. Cells may be None if not yet filled in.
    """
    with _conn() as c:
        clients = [_client_row_to_dict(r) for r in c.execute("""
            SELECT id, name, contact, email, phone, address, image
            FROM clients ORDER BY position, id
        """).fetchall()]
        ids = [cl["id"] for cl in clients]
        idx_of = {cid: i + 1 for i, cid in enumerate(ids)}
        n = len(ids) + 1
        matrix = [[None] * n for _ in range(n)]
        matrix[0][0] = 0.0

        rows = c.execute("""
            SELECT from_client_id, to_client_id, distance_km
            FROM client_distances
        """).fetchall()
        for r in rows:
            f, t = r["from_client_id"], r["to_client_id"]
            fi = 0 if f == 0 else idx_of.get(f)
            ti = 0 if t == 0 else idx_of.get(t)
            if fi is None or ti is None:
                continue
            matrix[fi][ti] = None if r["distance_km"] is None else float(r["distance_km"])

        return {"clients": clients, "ids": ids, "matrix": matrix, "n": n}


def set_distance(from_id: int, to_id: int, distance_km) -> None:
    val = None if distance_km is None else float(distance_km)
    with _conn() as c:
        c.execute("""
            INSERT INTO client_distances (from_client_id, to_client_id, distance_km)
            VALUES (?, ?, ?)
            ON CONFLICT(from_client_id, to_client_id)
            DO UPDATE SET distance_km = excluded.distance_km
        """, (from_id, to_id, val))


def set_distance_matrix(matrix, client_ids) -> None:
    """
    Bulk save.  matrix[i][j] maps to (client_ids[i-1] or 0, client_ids[j-1] or 0).
    """
    with _conn() as c:
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                fi = 0 if i == 0 else client_ids[i - 1]
                ti = 0 if j == 0 else client_ids[j - 1]
                v = None if val in (None, "", "None") else float(val)
                c.execute("""
                    INSERT INTO client_distances (from_client_id, to_client_id, distance_km)
                    VALUES (?, ?, ?)
                    ON CONFLICT(from_client_id, to_client_id)
                    DO UPDATE SET distance_km = excluded.distance_km
                """, (fi, ti, v))


# ───────────────────────────────────────────────────────────────────────
#  Suppliers  (many-to-many with products)
# ───────────────────────────────────────────────────────────────────────
def _supplier_base(row: sqlite3.Row) -> dict:
    return {
        "id":      row["id"],
        "name":    row["name"],
        "contact": row["contact"] or "",
        "email":   row["email"] or "",
        "phone":   row["phone"] or "",
        "address": row["address"] or "",
        "image":   row["image"],
    }


def get_suppliers() -> list[dict]:
    with _conn() as c:
        rows = c.execute("""
            SELECT id, name, contact, email, phone, address, image
            FROM suppliers ORDER BY position, id
        """).fetchall()
        suppliers = [_supplier_base(r) for r in rows]
        # bulk-load product links
        links = c.execute(
            "SELECT supplier_id, product_id FROM supplier_products"
        ).fetchall()
        by_supplier: dict[int, list[int]] = {}
        for l in links:
            by_supplier.setdefault(l["supplier_id"], []).append(l["product_id"])
        for s in suppliers:
            s["product_ids"] = by_supplier.get(s["id"], [])
        return suppliers


def get_supplier(sid: int) -> dict | None:
    with _conn() as c:
        row = c.execute("""
            SELECT id, name, contact, email, phone, address, image
            FROM suppliers WHERE id = ?
        """, (sid,)).fetchone()
        if not row:
            return None
        s = _supplier_base(row)
        s["product_ids"] = [
            r["product_id"] for r in c.execute(
                "SELECT product_id FROM supplier_products WHERE supplier_id = ? "
                "ORDER BY product_id", (sid,)
            ).fetchall()
        ]
        return s


def _set_supplier_products(c: sqlite3.Connection, sid: int, product_ids) -> None:
    c.execute("DELETE FROM supplier_products WHERE supplier_id = ?", (sid,))
    seen: set[int] = set()
    for pid in (product_ids or []):
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            continue
        if pid in seen:
            continue
        seen.add(pid)
        # only link to products that actually exist
        exists = c.execute("SELECT 1 FROM products WHERE id = ?", (pid,)).fetchone()
        if exists:
            c.execute(
                "INSERT OR IGNORE INTO supplier_products (supplier_id, product_id) "
                "VALUES (?, ?)", (sid, pid)
            )


def create_supplier(name, contact="", email="", phone="",
                    address="", image=None, product_ids=None) -> dict:
    name = str(name or "").strip()
    if not name:
        raise ValueError("Supplier name is required.")
    with _conn() as c:
        max_pos = c.execute(
            "SELECT COALESCE(MAX(position), -1) AS p FROM suppliers"
        ).fetchone()["p"]
        cur = c.execute("""
            INSERT INTO suppliers (name, contact, email, phone, address, image, position)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, contact or "", email or "", phone or "",
              address or "", image, max_pos + 1))
        sid = cur.lastrowid
        _set_supplier_products(c, sid, product_ids)
    return get_supplier(sid)


def update_supplier(sid, name, contact="", email="", phone="",
                    address="", image=None, product_ids=None) -> dict | None:
    name = str(name or "").strip()
    if not name:
        raise ValueError("Supplier name is required.")
    with _conn() as c:
        row = c.execute("SELECT id FROM suppliers WHERE id = ?", (sid,)).fetchone()
        if not row:
            return None
        c.execute("""
            UPDATE suppliers
               SET name = ?, contact = ?, email = ?, phone = ?, address = ?,
                   image = ?, updated_at = datetime('now')
             WHERE id = ?
        """, (name, contact or "", email or "", phone or "",
              address or "", image, sid))
        _set_supplier_products(c, sid, product_ids)
    return get_supplier(sid)


def delete_supplier(sid: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM supplier_products WHERE supplier_id = ?", (sid,))
        c.execute("DELETE FROM suppliers WHERE id = ?", (sid,))


# ───────────────────────────────────────────────────────────────────────
#  Employees  (+ shift log)
# ───────────────────────────────────────────────────────────────────────
def _calc_age(birthday) -> int | None:
    """
    Calculate age in whole years from a birthday.

    Accepts:
      - "YYYY-MM-DD" (HTML date input format)
      - a date or datetime object
      - None / "" / garbage -> returns None
    """
    if birthday is None:
        return None

    if isinstance(birthday, datetime):
        bd = birthday.date()
    elif isinstance(birthday, date):
        bd = birthday
    else:
        bd_str = str(birthday).strip()
        if not bd_str:
            return None
        try:
            bd = datetime.strptime(bd_str[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None

    today = date.today()
    years = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    # Sanity guard against typos like 1900-… or a future date
    if years < 0 or years > 150:
        return None
    return years


def refresh_all_ages() -> int:
    """
    Recompute `age` for every employee from their birthday.
    Returns the number of rows whose age changed.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT id, age, birthday FROM employees"
        ).fetchall()
        changed = 0
        for r in rows:
            new_age = _calc_age(r["birthday"])
            if new_age != r["age"]:
                c.execute(
                    "UPDATE employees SET age = ? WHERE id = ?",
                    (new_age, r["id"]),
                )
                changed += 1
        return changed


def _employee_base(row: sqlite3.Row) -> dict:
    return {
        "id":       row["id"],
        "name":     row["name"],
        "age":      row["age"],
        "sex":      row["sex"] or "",
        "birthday": row["birthday"],
        "email":    row["email"] or "",
        "phone":    row["phone"] or "",
        "address":  row["address"] or "",
        "image":    row["image"],
        "pay_type": row["pay_type"] or "hourly",
        "rate":     float(row["rate"] or 0),
        "pay_day":  row["pay_day"],
    }


def _employee_with_shifts(c: sqlite3.Connection, eid: int) -> dict | None:
    row = c.execute("""
        SELECT id, name, age, sex, birthday, email, phone, address, image,
               pay_type, rate, pay_day
        FROM employees WHERE id = ?
    """, (eid,)).fetchone()
    if not row:
        return None
    e = _employee_base(row)
    e["shifts"] = [
        dict(r) for r in c.execute("""
            SELECT id, date, hours FROM employee_shifts
            WHERE employee_id = ? ORDER BY date DESC, id DESC
        """, (eid,)).fetchall()
    ]
    return e


def get_employees() -> list[dict]:
    with _conn() as c:
        rows = c.execute("""
            SELECT id, name, age, sex, birthday, email, phone, address, image,
                   pay_type, rate, pay_day
            FROM employees ORDER BY position, id
        """).fetchall()
        employees = [_employee_base(r) for r in rows]
        # bulk-load shifts
        shifts = c.execute(
            "SELECT id, employee_id, date, hours FROM employee_shifts "
            "ORDER BY date DESC, id DESC"
        ).fetchall()
        by_emp: dict[int, list[dict]] = {}
        for s in shifts:
            by_emp.setdefault(s["employee_id"], []).append(
                {"id": s["id"], "date": s["date"], "hours": s["hours"]}
            )
        for e in employees:
            e["shifts"] = by_emp.get(e["id"], [])
        return employees


def get_employee(eid: int) -> dict | None:
    with _conn() as c:
        return _employee_with_shifts(c, eid)


def _norm_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _norm_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _norm_pay_day(v):
    d = _norm_int(v)
    if d is None:
        return None
    return d if 1 <= d <= 31 else None


def create_employee(name, age=None, sex="", birthday=None,
                    email="", phone="", address="", image=None,
                    pay_type="hourly", rate=0.0, pay_day=None) -> dict:
    name = str(name or "").strip()
    if not name:
        raise ValueError("Employee name is required.")
    sex = str(sex or "").strip().lower()
    pay_type = str(pay_type or "hourly").strip().lower()
    if pay_type not in ("hourly", "daily"):
        pay_type = "hourly"

    # derive age from birthday when possible
    computed_age = _calc_age(birthday)
    final_age = computed_age if computed_age is not None else _norm_int(age)

    with _conn() as c:
        max_pos = c.execute(
            "SELECT COALESCE(MAX(position), -1) AS p FROM employees"
        ).fetchone()["p"]
        cur = c.execute("""
            INSERT INTO employees
                (name, age, sex, birthday, email, phone, address, image,
                 pay_type, rate, pay_day, position)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, final_age, sex, birthday or None,
              email or "", phone or "", address or "", image,
              pay_type, _norm_float(rate), _norm_pay_day(pay_day),
              max_pos + 1))
        eid = cur.lastrowid
    return get_employee(eid)


def update_employee(eid, name, age=None, sex="", birthday=None,
                    email="", phone="", address="", image=None,
                    pay_type="hourly", rate=0.0, pay_day=None,
                    shifts=None) -> dict | None:
    """
    Update an employee. If `shifts` is provided (a list of
    {"id": <int|None>, "date": "YYYY-MM-DD", "hours": <float>}), the shift
    log is reconciled: existing ids are updated, new entries are inserted,
    and any pre-existing rows not in the list are removed.
    Pass shifts=None to leave the shift log untouched.
    """
    name = str(name or "").strip()
    if not name:
        raise ValueError("Employee name is required.")
    sex = str(sex or "").strip().lower()
    pay_type = str(pay_type or "hourly").strip().lower()
    if pay_type not in ("hourly", "daily"):
        pay_type = "hourly"

    # derive age from birthday when possible
    computed_age = _calc_age(birthday)
    final_age = computed_age if computed_age is not None else _norm_int(age)

    with _conn() as c:
        row = c.execute("SELECT id FROM employees WHERE id = ?", (eid,)).fetchone()
        if not row:
            return None
        c.execute("""
            UPDATE employees
               SET name = ?, age = ?, sex = ?, birthday = ?,
                   email = ?, phone = ?, address = ?, image = ?,
                   pay_type = ?, rate = ?, pay_day = ?,
                   updated_at = datetime('now')
             WHERE id = ?
        """, (name, final_age, sex, birthday or None,
              email or "", phone or "", address or "", image,
              pay_type, _norm_float(rate), _norm_pay_day(pay_day), eid))

        if shifts is not None:
            existing_ids = {
                r["id"] for r in c.execute(
                    "SELECT id FROM employee_shifts WHERE employee_id = ?",
                    (eid,)
                ).fetchall()
            }
            kept: set[int] = set()
            for s in shifts:
                sid = s.get("id")
                date = str(s.get("date") or "").strip()
                if not date:
                    continue
                hours = max(0.0, _norm_float(s.get("hours"), 1.0) or 1.0)
                if sid and sid in existing_ids:
                    c.execute(
                        "UPDATE employee_shifts SET date = ?, hours = ? WHERE id = ?",
                        (date, hours, sid)
                    )
                    kept.add(sid)
                else:
                    cur = c.execute(
                        "INSERT INTO employee_shifts (employee_id, date, hours) "
                        "VALUES (?, ?, ?)", (eid, date, hours)
                    )
                    kept.add(cur.lastrowid)
            for sid in (existing_ids - kept):
                c.execute("DELETE FROM employee_shifts WHERE id = ?", (sid,))
    return get_employee(eid)


def delete_employee(eid: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM employee_shifts WHERE employee_id = ?", (eid,))
        c.execute("DELETE FROM employees WHERE id = ?", (eid,))
