#!/usr/bin/env python3
"""
Inventory Management Calculation Engine
========================================
Complete Python program for:
  - ABC Analysis
  - XYZ Analysis
  - Safety Stock (SS)
  - Reorder Point (RoP)
  - Economic Order Quantity (EOQ)
  - Combined ABC-XYZ Classification

Supports CSV import and manual data entry.
Automatically detects demand frequency (daily/weekly/monthly/yearly) from
the Date column when importing from CSV.
"""

import sys
import math
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

SERVICE_LEVEL_Z_MAP: Dict[float, float] = {
    80.0: 0.842,
    85.0: 1.036,
    90.0: 1.282,
    95.0: 1.645,
    97.5: 1.960,
    98.0: 2.054,
    99.0: 2.326,
    99.5: 2.576,
    99.9: 3.090,
}

DAYS_PER_WEEK: float = 7.0
DAYS_PER_MONTH: float = 30.4375
DAYS_PER_YEAR: float = 365.0

DEFAULT_ABC_THRESHOLDS: Dict[str, float] = {"A": 0.80, "B": 0.95, "C": 1.00}
DEFAULT_XYZ_THRESHOLDS: Dict[str, float] = {"X": 0.50, "Y": 1.00, "Z": float("inf")}

# Median day-gap between observations -> demand frequency label
_INTERVAL_BOUNDS: List[Tuple[float, str]] = [
    (2.0,          "daily"),    # 1–2 days apart
    (10.0,         "weekly"),   # 3–10 days apart
    (45.0,         "monthly"),  # 11–45 days apart
    (float("inf"), "yearly"),   # >45 days apart
]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ProductInput:
    """Raw input data for a single product."""
    product_id: str
    product_name: str
    demand_history: List[float]
    unit_cost: float
    lead_time_days: float
    ordering_cost: float
    holding_cost_rate: float


@dataclass
class ProductResult:
    """Calculated results for a single product."""
    product_id: str
    product_name: str
    mean_demand: float
    demand_stddev: float
    annual_demand: float
    unit_cost: float
    annual_consumption_value: float
    cumulative_acv_percentage: float
    abc_class: str
    cv: float
    xyz_class: str
    combined_class: str
    lead_time_days: float
    lead_time_periods: float
    service_level: float
    z_value: float
    safety_stock: float
    lead_time_demand: float
    reorder_point: float
    ordering_cost: float
    holding_cost_rate: float
    annual_holding_cost_per_unit: float
    eoq: Optional[float]


# ---------------------------------------------------------------------------
# Date-Based Frequency Detection
# ---------------------------------------------------------------------------

def classify_interval(days: float) -> str:
    """Map a median day-gap to a demand frequency label."""
    for bound, label in _INTERVAL_BOUNDS:
        if days <= bound:
            return label
    return "yearly"


def detect_period_from_dates(dates) -> Optional[str]:
    """
    Infer demand frequency from a sequence of observation dates.
    Uses the median gap between consecutive unique dates so a few
    missing/extra periods don't skew the answer.
    """
    if dates is None:
        return None

    parsed = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    if len(parsed) < 2:
        return None

    unique_days = np.sort(np.array(parsed.dt.normalize().unique(),
                                   dtype="datetime64[D]"))
    diffs = np.diff(unique_days).astype(float)
    if diffs.size == 0:
        return None

    return classify_interval(float(np.median(diffs)))


def detect_period_from_products(product_dates: Dict[str, list]) -> Optional[str]:
    """
    Vote across products; returns the most common detected period.
    Ties are broken by first-seen order.
    """
    votes: Counter = Counter()
    for dates in product_dates.values():
        p = detect_period_from_dates(dates)
        if p:
            votes[p] += 1

    if not votes:
        return None
    return votes.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Input / Data Loading
# ---------------------------------------------------------------------------

def normalize_percentage(value) -> float:
    """
    Accept a percentage as 0.25 or 25 and return 0.25.
    Warns if the value looks like a raw percentage (>1.5).
    """
    try:
        v = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Cannot convert '{value}' to a number.")

    if v < 0:
        raise ValueError(f"Percentage cannot be negative (got {v}).")

    if v > 1.5:
        # Likely entered as 25 instead of 0.25
        return v / 100.0
    return v


def parse_demand_history(demand_str: str) -> List[float]:
    """Parse a comma-separated string of demand values."""
    values = [v.strip() for v in demand_str.split(",") if v.strip()]
    result = []
    for v in values:
        try:
            result.append(float(v))
        except ValueError:
            raise ValueError(f"Invalid demand value: '{v}'")
    return result


def load_csv(filepath: str) -> Tuple[List["ProductInput"], Optional[str]]:
    """
    Load product data from a CSV file.

    Expected CSV columns:
        Product_ID, Product_Name, Date, Demand, Unit_Cost,
        Lead_Time_Days, Ordering_Cost, Holding_Cost_Rate

    Each row is one historical demand observation.

    Returns:
        (list_of_products, detected_period_or_None)
        detected_period is one of {"daily", "weekly", "monthly", "yearly"}.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(filepath, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    required_cols = [
        "Product_ID", "Product_Name", "Demand",
        "Unit_Cost", "Lead_Time_Days", "Ordering_Cost", "Holding_Cost_Rate"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    has_date_col = "Date" in df.columns

    products: Dict[str, ProductInput] = {}
    product_dates: Dict[str, list] = {}
    errors: List[str] = []

    for idx, row in df.iterrows():
        row_num = idx + 2  # 1-based, +header
        pid = str(row.get("Product_ID", "")).strip()
        if not pid:
            errors.append(f"Row {row_num}: Missing Product_ID")
            continue

        try:
            demand_val = float(row["Demand"])
        except (ValueError, TypeError):
            errors.append(f"Row {row_num} (Product {pid}): Invalid Demand '{row['Demand']}'")
            continue

        if demand_val < 0:
            errors.append(f"Row {row_num} (Product {pid}): Negative demand {demand_val}")
            continue

        # Track date for frequency detection (only for valid demand rows)
        if has_date_col:
            product_dates.setdefault(pid, []).append(row.get("Date"))

        if pid not in products:
            # First time seeing this product — parse static fields
            try:
                unit_cost = float(row["Unit_Cost"])
                if unit_cost < 0:
                    raise ValueError("negative")
            except (ValueError, TypeError):
                errors.append(f"Row {row_num} (Product {pid}): Invalid Unit_Cost '{row['Unit_Cost']}'")
                continue

            try:
                lead_time = float(row["Lead_Time_Days"])
                if lead_time < 0:
                    raise ValueError("negative")
            except (ValueError, TypeError):
                errors.append(f"Row {row_num} (Product {pid}): Invalid Lead_Time_Days '{row['Lead_Time_Days']}'")
                continue

            try:
                ordering_cost = float(row["Ordering_Cost"])
                if ordering_cost <= 0:
                    raise ValueError("non-positive")
            except (ValueError, TypeError):
                errors.append(f"Row {row_num} (Product {pid}): Invalid Ordering_Cost '{row['Ordering_Cost']}' (must be > 0)")
                continue

            try:
                holding_rate = normalize_percentage(row["Holding_Cost_Rate"])
                if holding_rate <= 0:
                    raise ValueError("non-positive")
            except (ValueError, TypeError) as e:
                errors.append(f"Row {row_num} (Product {pid}): Invalid Holding_Cost_Rate '{row['Holding_Cost_Rate']}' — {e}")
                continue

            products[pid] = ProductInput(
                product_id=pid,
                product_name=str(row.get("Product_Name", pid)).strip(),
                demand_history=[demand_val],
                unit_cost=unit_cost,
                lead_time_days=lead_time,
                ordering_cost=ordering_cost,
                holding_cost_rate=holding_rate,
            )
        else:
            products[pid].demand_history.append(demand_val)

    if errors:
        print("\n[CSV Validation Errors]")
        for e in errors:
            print(f"  • {e}")
        print()
        if not products:
            raise ValueError("No valid products could be loaded from the CSV.")
        choice = input(f"{len(errors)} error(s) found. Continue with valid products? (y/n): ").strip().lower()
        if choice not in ("y", "yes"):
            raise ValueError("Import aborted by user.")

    detected = detect_period_from_products(product_dates) if has_date_col else None
    return list(products.values()), detected


def manual_entry() -> List[ProductInput]:
    """Interactive manual data entry for multiple products."""
    print("\n" + "=" * 50)
    print("      MANUAL DATA ENTRY MODE")
    print("=" * 50)

    while True:
        try:
            n = int(input("\nNumber of products to enter: ").strip())
            if n <= 0:
                print("Please enter a positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a positive integer.")

    products: List[ProductInput] = []

    for i in range(1, n + 1):
        print(f"\n--- Product {i} of {n} ---")

        pid = input("Product ID: ").strip()
        while not pid:
            print("Product ID cannot be empty.")
            pid = input("Product ID: ").strip()

        pname = input("Product Name: ").strip() or pid

        while True:
            demand_str = input("Historical demand values (comma-separated): ").strip()
            try:
                demand_history = parse_demand_history(demand_str)
                if not demand_history:
                    print("At least one demand value is required.")
                    continue
                if any(d < 0 for d in demand_history):
                    print("Demand values cannot be negative.")
                    continue
                break
            except ValueError as e:
                print(e)

        while True:
            try:
                unit_cost = float(input("Unit cost: ").strip())
                if unit_cost < 0:
                    print("Unit cost cannot be negative.")
                    continue
                break
            except ValueError:
                print("Invalid number. Please try again.")

        while True:
            try:
                lead_time = float(input("Lead time (days): ").strip())
                if lead_time < 0:
                    print("Lead time cannot be negative.")
                    continue
                break
            except ValueError:
                print("Invalid number. Please try again.")

        while True:
            try:
                ordering_cost = float(input("Ordering cost per order: ").strip())
                if ordering_cost <= 0:
                    print("Ordering cost must be positive.")
                    continue
                break
            except ValueError:
                print("Invalid number. Please try again.")

        while True:
            rate_str = input("Holding cost rate (e.g., 0.25 or 25 for 25%): ").strip()
            try:
                holding_rate = normalize_percentage(rate_str)
                if holding_rate <= 0:
                    print("Holding cost rate must be positive.")
                    continue
                break
            except ValueError as e:
                print(e)

        products.append(ProductInput(
            product_id=pid,
            product_name=pname,
            demand_history=demand_history,
            unit_cost=unit_cost,
            lead_time_days=lead_time,
            ordering_cost=ordering_cost,
            holding_cost_rate=holding_rate,
        ))

    return products


# ---------------------------------------------------------------------------
# Calculation Functions
# ---------------------------------------------------------------------------

def get_periods_per_year(demand_period: str) -> float:
    """Return the number of demand periods in one year."""
    mapping = {
        "daily":   DAYS_PER_YEAR,   # 365
        "weekly":  52.0,
        "monthly": 12.0,
        "yearly":  1.0,
    }
    dp = demand_period.lower().strip()
    if dp not in mapping:
        raise ValueError(
            f"Invalid demand period: '{demand_period}'. "
            "Use Daily, Weekly, Monthly, or Yearly."
        )
    return mapping[dp]


def convert_lead_time(lead_time_days: float, demand_period: str) -> float:
    """Convert lead time in days to the selected demand period."""
    dp = demand_period.lower().strip()
    if dp == "daily":
        return lead_time_days
    elif dp == "weekly":
        return lead_time_days / DAYS_PER_WEEK
    elif dp == "monthly":
        return lead_time_days / DAYS_PER_MONTH
    elif dp == "yearly":
        return lead_time_days / DAYS_PER_YEAR
    else:
        raise ValueError(f"Invalid demand period: '{demand_period}'")


def calculate_demand_statistics(demand_history: List[float]) -> Tuple[float, float]:
    """
    Calculate mean demand and sample standard deviation.
    Returns (mean, stddev).
    """
    arr = np.array(demand_history, dtype=float)
    mean_demand = float(np.mean(arr))

    if len(arr) == 1:
        stddev = 0.0
    else:
        stddev = float(np.std(arr, ddof=1))

    return mean_demand, stddev


def calculate_annual_demand(mean_demand: float, demand_period: str) -> float:
    """Convert mean demand per period to annual demand."""
    periods = get_periods_per_year(demand_period)
    return mean_demand * periods


def calculate_abc(
    products: List[ProductInput],
    demand_period: str,
    a_threshold: float,
    b_threshold: float,
) -> List[Tuple[ProductInput, float, float, str]]:
    """
    Perform ABC classification.
    Returns list of tuples:
        (product, annual_demand, annual_consumption_value, abc_class)
    sorted by ACV descending.
    """
    items: List[Tuple[ProductInput, float, float]] = []
    for p in products:
        mean_demand, _ = calculate_demand_statistics(p.demand_history)
        annual_demand = calculate_annual_demand(mean_demand, demand_period)
        acv = annual_demand * p.unit_cost
        items.append((p, annual_demand, acv))

    items.sort(key=lambda x: x[2], reverse=True)

    total_acv = sum(acv for _, _, acv in items)
    if total_acv == 0:
        return [(p, ad, acv, "C") for p, ad, acv in items]

    cumulative = 0.0
    results: List[Tuple[ProductInput, float, float, str]] = []

    for p, annual_demand, acv in items:
        cumulative += acv
        cum_pct = cumulative / total_acv

        if cum_pct <= a_threshold:
            abc_class = "A"
        elif cum_pct <= b_threshold:
            abc_class = "B"
        else:
            abc_class = "C"

        results.append((p, annual_demand, acv, abc_class))

    return results


def calculate_xyz(
    products: List[ProductInput],
    x_threshold: float,
    y_threshold: float,
) -> Dict[str, Tuple[float, str]]:
    """
    Perform XYZ classification.
    Returns dict mapping product_id -> (cv, xyz_class).
    """
    results: Dict[str, Tuple[float, str]] = {}

    for p in products:
        mean_demand, stddev = calculate_demand_statistics(p.demand_history)

        if mean_demand == 0:
            cv = float("inf") if stddev > 0 else 0.0
            xyz_class = "Z"
        else:
            cv = stddev / mean_demand
            if cv <= x_threshold:
                xyz_class = "X"
            elif cv <= y_threshold:
                xyz_class = "Y"
            else:
                xyz_class = "Z"

        results[p.product_id] = (cv, xyz_class)

    return results


def calculate_safety_stock(
    demand_stddev: float,
    lead_time_periods: float,
    z_value: float,
) -> float:
    """Calculate Safety Stock = Z * σd * sqrt(LT)."""
    if lead_time_periods <= 0:
        return 0.0
    return z_value * demand_stddev * math.sqrt(lead_time_periods)


def calculate_reorder_point(
    mean_demand: float,
    lead_time_periods: float,
    safety_stock: float,
) -> float:
    """Calculate Reorder Point = Mean Demand * LT + Safety Stock."""
    if lead_time_periods <= 0:
        return safety_stock
    return mean_demand * lead_time_periods + safety_stock


def calculate_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_per_unit: float,
) -> Optional[float]:
    """
    Calculate EOQ = sqrt((2 * D * S) / H).
    Returns None if any input is non-positive.
    """
    if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_per_unit <= 0:
        return None
    return math.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_calculations(
    products: List[ProductInput],
    demand_period: str,
    service_level: float,
    z_value: float,
    abc_thresholds: Dict[str, float],
    xyz_thresholds: Dict[str, float],
) -> List[ProductResult]:
    """
    Run the full calculation pipeline and return a list of ProductResult objects,
    sorted by Annual Consumption Value descending.
    """
    abc_results = calculate_abc(
        products,
        demand_period,
        abc_thresholds["A"],
        abc_thresholds["B"],
    )

    xyz_results = calculate_xyz(
        products,
        xyz_thresholds["X"],
        xyz_thresholds["Y"],
    )

    total_acv = sum(acv for _, _, acv, _ in abc_results)
    cumulative_acv = 0.0

    results: List[ProductResult] = []

    for p, annual_demand, acv, abc_class in abc_results:
        cumulative_acv += acv
        cum_pct = (cumulative_acv / total_acv * 100.0) if total_acv > 0 else 0.0

        mean_demand, demand_stddev = calculate_demand_statistics(p.demand_history)
        lead_time_periods = convert_lead_time(p.lead_time_days, demand_period)

        cv, xyz_class = xyz_results[p.product_id]
        combined_class = abc_class + xyz_class

        safety_stock = calculate_safety_stock(demand_stddev, lead_time_periods, z_value)
        lead_time_demand = mean_demand * lead_time_periods
        reorder_point = calculate_reorder_point(mean_demand, lead_time_periods, safety_stock)

        annual_holding_cost_per_unit = p.unit_cost * p.holding_cost_rate
        eoq = calculate_eoq(annual_demand, p.ordering_cost, annual_holding_cost_per_unit)

        results.append(ProductResult(
            product_id=p.product_id,
            product_name=p.product_name,
            mean_demand=mean_demand,
            demand_stddev=demand_stddev,
            annual_demand=annual_demand,
            unit_cost=p.unit_cost,
            annual_consumption_value=acv,
            cumulative_acv_percentage=cum_pct,
            abc_class=abc_class,
            cv=cv,
            xyz_class=xyz_class,
            combined_class=combined_class,
            lead_time_days=p.lead_time_days,
            lead_time_periods=lead_time_periods,
            service_level=service_level,
            z_value=z_value,
            safety_stock=safety_stock,
            lead_time_demand=lead_time_demand,
            reorder_point=reorder_point,
            ordering_cost=p.ordering_cost,
            holding_cost_rate=p.holding_cost_rate,
            annual_holding_cost_per_unit=annual_holding_cost_per_unit,
            eoq=eoq,
        ))

    return results


# ---------------------------------------------------------------------------
# Report & Export
# ---------------------------------------------------------------------------

def generate_report(results: List[ProductResult]) -> None:
    """Print a formatted inventory management report to stdout."""
    print("\n" + "=" * 80)
    print("           INVENTORY MANAGEMENT CALCULATION REPORT")
    print("=" * 80)

    n_products = len(results)
    total_acv = sum(r.annual_consumption_value for r in results)

    abc_counts = {"A": 0, "B": 0, "C": 0}
    xyz_counts = {"X": 0, "Y": 0, "Z": 0}
    combined_counts: Dict[str, int] = {}

    for r in results:
        abc_counts[r.abc_class] = abc_counts.get(r.abc_class, 0) + 1
        xyz_counts[r.xyz_class] = xyz_counts.get(r.xyz_class, 0) + 1
        combined_counts[r.combined_class] = combined_counts.get(r.combined_class, 0) + 1

    print(f"\n📊 SUMMARY")
    print(f"   Total Products:        {n_products}")
    print(f"   Total Annual ACV:      {total_acv:,.2f}")
    print(f"\n   ABC Breakdown:")
    for cls in ("A", "B", "C"):
        print(f"      {cls}: {abc_counts.get(cls, 0)}")
    print(f"\n   XYZ Breakdown:")
    for cls in ("X", "Y", "Z"):
        print(f"      {cls}: {xyz_counts.get(cls, 0)}")
    print(f"\n   Combined ABC-XYZ Breakdown:")
    for cls in sorted(combined_counts.keys()):
        print(f"      {cls}: {combined_counts[cls]}")

    print(f"\n{'Product':<12} {'ABC':<5} {'XYZ':<5} {'CV':<8} {'AnnualDmd':<12} {'SS':<10} {'RoP':<10} {'EOQ':<10}")
    print("-" * 80)
    for r in results:
        eoq_str = f"{r.eoq:,.1f}" if r.eoq is not None else "N/A"
        print(
            f"{r.product_id:<12} {r.abc_class:<5} {r.xyz_class:<5} "
            f"{r.cv:<8.4f} {r.annual_demand:<12.2f} {r.safety_stock:<10.2f} "
            f"{r.reorder_point:<10.2f} {eoq_str:<10}"
        )

    print("\n" + "=" * 80)


def export_results(results: List[ProductResult], filepath: str) -> None:
    """Export all calculated results to a CSV file."""
    rows = []
    for r in results:
        rows.append({
            "Product_ID": r.product_id,
            "Product_Name": r.product_name,
            "Mean_Demand": round(r.mean_demand, 4),
            "Demand_StdDev": round(r.demand_stddev, 4),
            "Annual_Demand": round(r.annual_demand, 2),
            "Unit_Cost": r.unit_cost,
            "Annual_Consumption_Value": round(r.annual_consumption_value, 2),
            "Cumulative_ACV_Percentage": round(r.cumulative_acv_percentage, 4),
            "ABC_Class": r.abc_class,
            "CV": round(r.cv, 6) if math.isfinite(r.cv) else "INF",
            "XYZ_Class": r.xyz_class,
            "Combined_Class": r.combined_class,
            "Lead_Time_Days": r.lead_time_days,
            "Lead_Time_Periods": round(r.lead_time_periods, 4),
            "Service_Level": r.service_level,
            "Z_Value": r.z_value,
            "Safety_Stock": round(r.safety_stock, 2),
            "Lead_Time_Demand": round(r.lead_time_demand, 2),
            "Reorder_Point": round(r.reorder_point, 2),
            "Ordering_Cost": r.ordering_cost,
            "Holding_Cost_Rate": r.holding_cost_rate,
            "Annual_Holding_Cost_Per_Unit": round(r.annual_holding_cost_per_unit, 4),
            "EOQ": round(r.eoq, 2) if r.eoq is not None else None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"\n✅ Results exported to: {filepath}")


# ---------------------------------------------------------------------------
# User Interaction Helpers
# ---------------------------------------------------------------------------

def select_demand_period() -> str:
    """Prompt user for demand frequency."""
    print("\nSelect demand frequency:")
    print("  1. Daily")
    print("  2. Weekly")
    print("  3. Monthly")
    print("  4. Yearly")
    while True:
        choice = input("Enter choice (1/2/3/4): ").strip()
        mapping = {"1": "daily", "2": "weekly", "3": "monthly", "4": "yearly"}
        if choice in mapping:
            return mapping[choice]
        print("Invalid choice. Please enter 1, 2, 3, or 4.")


def select_service_level() -> Tuple[float, float]:
    """Prompt user for service level and return (service_level_pct, z_value)."""
    print("\nSelect service level:")
    for i, (sl, z) in enumerate(SERVICE_LEVEL_Z_MAP.items(), 1):
        print(f"  {i}. {sl}%  (Z = {z})")
    print(f"  {len(SERVICE_LEVEL_Z_MAP) + 1}. Enter custom Z value")

    while True:
        choice = input(f"Enter choice (1-{len(SERVICE_LEVEL_Z_MAP) + 1}): ").strip()
        try:
            idx = int(choice)
        except ValueError:
            print("Invalid choice.")
            continue

        if 1 <= idx <= len(SERVICE_LEVEL_Z_MAP):
            sl = list(SERVICE_LEVEL_Z_MAP.keys())[idx - 1]
            return sl, SERVICE_LEVEL_Z_MAP[sl]
        elif idx == len(SERVICE_LEVEL_Z_MAP) + 1:
            while True:
                z_str = input("Enter custom Z value: ").strip()
                try:
                    z = float(z_str)
                    if z <= 0:
                        print("Z value must be positive.")
                        continue
                    return 0.0, z  # 0.0 indicates custom
                except ValueError:
                    print("Invalid number.")
        else:
            print("Invalid choice.")


def select_abc_thresholds() -> Dict[str, float]:
    """Prompt user for ABC classification thresholds."""
    print(f"\nDefault ABC thresholds: A=0–80%, B=80–95%, C=95–100%")
    choice = input("Use default thresholds? (y/n): ").strip().lower()
    if choice in ("y", "yes", ""):
        return DEFAULT_ABC_THRESHOLDS.copy()

    while True:
        try:
            a_thresh = float(input("A threshold (cumulative %, e.g., 0.80): ").strip())
            b_thresh = float(input("B threshold (cumulative %, e.g., 0.95): ").strip())
            if not (0 < a_thresh < b_thresh <= 1.0):
                print("Invalid: must have 0 < A < B <= 1.0")
                continue
            return {"A": a_thresh, "B": b_thresh, "C": 1.0}
        except ValueError:
            print("Invalid number. Please try again.")


def select_xyz_thresholds() -> Dict[str, float]:
    """Prompt user for XYZ classification thresholds."""
    print(f"\nDefault XYZ thresholds: X=CV≤0.50, Y=0.50<CV≤1.00, Z=CV>1.00")
    choice = input("Use default thresholds? (y/n): ").strip().lower()
    if choice in ("y", "yes", ""):
        return DEFAULT_XYZ_THRESHOLDS.copy()

    while True:
        try:
            x_thresh = float(input("X threshold (max CV for X): ").strip())
            y_thresh = float(input("Y threshold (max CV for Y): ").strip())
            if not (0 <= x_thresh < y_thresh):
                print("Invalid: must have 0 <= X < Y")
                continue
            return {"X": x_thresh, "Y": y_thresh, "Z": float("inf")}
        except ValueError:
            print("Invalid number. Please try again.")


def select_input_mode() -> Tuple[str, Optional[str]]:
    """Prompt user for input mode and return (mode, filepath_or_none)."""
    print("\nSelect input mode:")
    print("  1. Import from CSV file")
    print("  2. Manual data entry")
    while True:
        choice = input("Enter choice (1/2): ").strip()
        if choice == "1":
            filepath = input("Enter CSV file path: ").strip()
            return "csv", filepath
        elif choice == "2":
            return "manual", None
        print("Invalid choice. Please enter 1 or 2.")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("   INVENTORY MANAGEMENT CALCULATION ENGINE")
    print("=" * 60)

    # Input mode
    mode, filepath = select_input_mode()

    # ------------------------------------------------------------------
    # Demand period: auto-detect for CSV, prompt for manual
    # ------------------------------------------------------------------
    if mode == "csv":
        products, detected_period = load_csv(filepath)

        if detected_period:
            print(f"\n🔎 Auto-detected demand frequency from Date column: {detected_period}")
            use = input("Use this? (y/n): ").strip().lower()
            if use in ("y", "yes", ""):
                demand_period = detected_period
            else:
                demand_period = select_demand_period()
        else:
            print("\n⚠️  Could not infer frequency from dates "
                  "(missing Date column or insufficient rows).")
            demand_period = select_demand_period()
    else:
        demand_period = select_demand_period()
        products = manual_entry()

    # Service level
    service_level, z_value = select_service_level()

    # Thresholds
    abc_thresholds = select_abc_thresholds()
    xyz_thresholds = select_xyz_thresholds()

    if not products:
        print("No products to process. Exiting.")
        sys.exit(0)

    # Run calculations
    results = run_calculations(
        products,
        demand_period,
        service_level,
        z_value,
        abc_thresholds,
        xyz_thresholds,
    )

    # Report
    generate_report(results)

    # Export
    export_path = input("\nEnter output CSV file path (or press Enter for 'inventory_results.csv'): ").strip()
    if not export_path:
        export_path = "inventory_results.csv"
    export_results(results, export_path)

    print("\n✅ All calculations complete. Goodbye!\n")


if __name__ == "__main__":
    main()