import os
import math
import json
import time
import random 
import tempfile
from flask import Flask, request, jsonify, send_from_directory, render_template_string

import pandas as pd

# Import the FJSP solver from the solvers folder
from solvers.fjsp import FlexibleJobShopSolver

# ----- NEW: Import the inventory classifier from the solvers folder -----
from solvers.classifier import (
    SERVICE_LEVEL_Z_MAP,
    DEFAULT_ABC_THRESHOLDS,
    DEFAULT_XYZ_THRESHOLDS,
    ProductInput,
    detect_period_from_products,
    normalize_percentage,
    parse_demand_history,
    run_calculations,
)

# ----- NEW: Import the CatBoost DemandForecaster -----
from solvers.CatBoost import DemandForecaster

# ----- NEW: Shared SQLite layer -----
from database.db import (
    init_db,
    # Workstations
    get_workstations,
    get_workstation,
    replace_workstations,
    create_workstation,
    update_workstation,
    delete_workstation,
    clear_workstations,
    # Products
    get_products,
    get_product,
    create_product,
    update_product,
    delete_product,
    # Product demand history  (NEW)
    get_product_demand,
    save_product_demand,
    get_products_with_demand,
    upsert_product_with_demand,
    # Clients
    get_clients, get_client, create_client, update_client, delete_client,
    # Trucks
    get_trucks, get_truck, create_truck, update_truck, delete_truck,
    # Distances
    get_distance_matrix, set_distance, set_distance_matrix,
    # Suppliers
    get_suppliers, get_supplier, create_supplier, update_supplier, delete_supplier,
    # Employees
    get_employees, get_employee, create_employee, update_employee, delete_employee,
)

app = Flask(__name__)

# ----- NEW: make sure the shared DB exists on startup -----
init_db()

# Configuration
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), 'frontend')

# ---------- Global state for the demand forecaster ----------
forecaster = None          # will hold a DemandForecaster instance
forecaster_metrics = {}    # store last metrics
forecaster_summary = {}    # store data summary

# ---------- NEW: Service levels exposed to the stock.html template ----------
SERVICE_LEVELS = [
    {"value": float(sl), "z": z} for sl, z in SERVICE_LEVEL_Z_MAP.items()
]

# ---------- Routes for frontend pages ----------
@app.route('/')
def home():
    return send_from_directory(FRONTEND_DIR, 'home.html')

@app.route('/forecast')
def forecast():
    return send_from_directory(FRONTEND_DIR, 'forecast.html')

@app.route('/stock')
def stock():
    # stock.html contains Jinja markup ({{ service_levels }}), so it must be
    # rendered with render_template_string instead of send_from_directory.
    template_path = os.path.join(FRONTEND_DIR, 'stock.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    return render_template_string(template, service_levels=SERVICE_LEVELS)

@app.route('/vrp')
def vrp():
    return send_from_directory(FRONTEND_DIR, 'vrp.html')

@app.route('/scheduling')
def scheduling():
    return send_from_directory(FRONTEND_DIR, 'fjsp.html')

# ---------- FJSP Scheduling Endpoint ----------
@app.route('/api/schedule', methods=['POST'])
def schedule_fjsp():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON payload'}), 400

        workstations = data.get('workstations')
        machines_per_ws = data.get('machines_per_ws')
        job_routings = data.get('job_routings')
        processing_times_in = data.get('processing_times')
        generations = data.get('generations', 150)
        pop_size = data.get('pop_size', 200)
        time_limit = data.get('time_limit')  # None if not provided

        if not workstations or not machines_per_ws or not job_routings or processing_times_in is None:
            return jsonify({'error': 'Missing required fields'}), 400

        # Convert processing times to (ws, job) -> duration
        p_times = {}
        for key, val in processing_times_in.items():
            ws_str, job_str = key.split(',')
            p_times[(int(ws_str), int(job_str))] = float(val)

        solver = FlexibleJobShopSolver(
            workstations=workstations,
            machines_per_ws=machines_per_ws,
            job_routings=job_routings,
            processing_times=p_times
        )

        start_time = time.time()
        best_ind, _ = solver.run_ga(
            pop_size=pop_size,
            ngen=generations,
            base_mut_prob=0.3,
            max_mut_prob=0.9,
            cx_prob=0.8,
            elite_size=3,
            stagnation_limit=25,
            diversity_threshold=0.15,
            verbose=False,
            time_limit=time_limit
        )
        makespan, schedule = solver.decode_chromosome(best_ind)
        runtime = time.time() - start_time

        # Build operations list for response
        ops_list = []
        for (ws, m), ops in schedule.items():
            for (start_t, end_t, job, op_idx) in ops:
                dur = p_times.get((ws, job), end_t - start_t)
                ops_list.append({
                    'workstation': ws,
                    'machine': m,
                    'start': start_t,
                    'end': end_t,
                    'job': job,
                    'op_index': op_idx,
                    'duration': dur
                })

        total_machine_time = sum(op['duration'] for op in ops_list)
        total_machine_capacity = sum(machines_per_ws) * makespan
        utilization = total_machine_time / total_machine_capacity if total_machine_capacity > 0 else 0

        response = {
            'makespan': makespan,
            'operations': ops_list,
            'runtime': runtime,
            'utilization': min(utilization, 1.0),
            'workstations': workstations,
            'machines_per_ws': machines_per_ws
        }
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================
#                  VRP SOLVER (integrated from vrp.py)
# ============================================================

def create_initial_solution(num_customers):
    """Return a random permutation of customer indices (1..num_customers)."""
    return random.sample(range(1, num_customers + 1), num_customers)

def evaluate(solution, customers, vehicles, dist_matrix):
    """
    Evaluate a solution (list of customer indices).
    Returns total_distance, total_cost, and route_details.
    """
    total_distance = 0.0
    total_cost = 0.0
    routes = []          # each element: (route_list, vehicle_idx, weight, volume)
    route = []
    route_weight = 0.0
    route_volume = 0.0
    vehicle_idx = 0

    for customer_idx in solution:
        # customers is a list of dicts; customer_idx is 1-based
        weight = customers[customer_idx - 1]['weight']
        volume = customers[customer_idx - 1]['volume']
        if (route_weight + weight <= vehicles[vehicle_idx]['max_weight'] and
            route_volume + volume <= vehicles[vehicle_idx]['max_volume']):
            route.append(customer_idx)
            route_weight += weight
            route_volume += volume
        else:
            # store current route and start a new one with current customer
            routes.append((route, vehicle_idx, route_weight, route_volume))
            route = [customer_idx]
            route_weight = weight
            route_volume = volume
            vehicle_idx = (vehicle_idx + 1) % len(vehicles)

    if route:
        routes.append((route, vehicle_idx, route_weight, route_volume))

    # Compute distance and cost for each route
    route_details = []
    for route, vi, w, vol in routes:
        # distance from depot to first customer
        d = dist_matrix[0][route[0]]
        # between customers
        for i in range(len(route) - 1):
            d += dist_matrix[route[i]][route[i + 1]]
        # back to depot
        d += dist_matrix[route[-1]][0]
        cost = d * vehicles[vi]['cost_per_km']
        total_distance += d
        total_cost += cost
        route_details.append({
            'route': route,
            'vehicle_idx': vi,
            'vehicle_name': vehicles[vi]['vehicle'],
            'distance': d,
            'cost': cost,
            'weight': w,
            'volume': vol
        })

    return total_distance, total_cost, route_details

# ---------- GA operators ----------
def select(population, fitnesses, k=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), k)
        selected.append(min(tournament, key=lambda x: x[1])[0])
    return selected

def crossover(parent1, parent2):
    size = len(parent1)
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    child1 = [None] * size
    child2 = [None] * size
    child1[cxpoint1:cxpoint2] = parent1[cxpoint1:cxpoint2]
    child2[cxpoint1:cxpoint2] = parent2[cxpoint1:cxpoint2]
    fill_child(child1, parent2, cxpoint2, size)
    fill_child(child2, parent1, cxpoint2, size)
    return child1, child2

def fill_child(child, parent, start, size):
    pos = start
    for i in range(size):
        if parent[i] not in child:
            while child[pos] is not None:
                pos = (pos + 1) % size
            child[pos] = parent[i]

def mutate(solution, mutpb):
    for i in range(len(solution)):
        if random.random() < mutpb:
            j = random.randint(0, len(solution) - 1)
            solution[i], solution[j] = solution[j], solution[i]

def genetic_algorithm(customers, vehicles, dist_matrix, pop_size, generations, cxpb, mutpb):
    num_customers = len(customers)
    population = [create_initial_solution(num_customers) for _ in range(pop_size)]
    fitnesses = [evaluate(ind, customers, vehicles, dist_matrix)[1] for ind in population]

    for _ in range(generations):
        selected = select(population, fitnesses)
        next_population = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected) and random.random() < cxpb:
                child1, child2 = crossover(selected[i], selected[i + 1])
            else:
                child1, child2 = selected[i], selected[i + 1] if i + 1 < len(selected) else selected[i]
            next_population.append(child1)
            next_population.append(child2)
        for ind in next_population:
            mutate(ind, mutpb)
        population = next_population[:pop_size]
        fitnesses = [evaluate(ind, customers, vehicles, dist_matrix)[1] for ind in population]

    best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
    best_solution = population[best_idx]
    best_dist, best_cost, best_route_details = evaluate(best_solution, customers, vehicles, dist_matrix)
    return best_solution, best_dist, best_cost, best_route_details

# ---------- VNS ----------
def local_search(solution, customers, vehicles, dist_matrix):
    best_solution = solution.copy()
    best_dist, best_cost, _ = evaluate(best_solution, customers, vehicles, dist_matrix)
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            new_solution = solution.copy()
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            new_dist, new_cost, _ = evaluate(new_solution, customers, vehicles, dist_matrix)
            if new_cost < best_cost:
                best_solution, best_dist, best_cost = new_solution, new_dist, new_cost
    return best_solution, best_dist, best_cost

def shake(solution, k):
    new_solution = solution.copy()
    for _ in range(k):
        i, j = random.sample(range(len(solution)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

def variable_neighbourhood_search(customers, vehicles, dist_matrix, max_iterations, k_max):
    num_customers = len(customers)
    best_solution = create_initial_solution(num_customers)
    best_dist, best_cost, _ = evaluate(best_solution, customers, vehicles, dist_matrix)
    iteration = 0
    while iteration < max_iterations:
        k = 1
        while k <= k_max:
            new_solution = shake(best_solution, k)
            new_solution, new_dist, new_cost = local_search(new_solution, customers, vehicles, dist_matrix)
            if new_cost < best_cost:
                best_solution, best_dist, best_cost = new_solution, new_dist, new_cost
                k = 1
            else:
                k += 1
        iteration += 1
    _, _, route_details = evaluate(best_solution, customers, vehicles, dist_matrix)
    return best_solution, best_dist, best_cost, route_details

# ---------- Wrapper for API ----------
def solve_vrp(customers, vehicles, dist_matrix, algorithm='ga', params=None):
    """
    customers: list of dicts with keys 'customer', 'weight', 'volume'
    vehicles: list of dicts with keys 'vehicle', 'max_weight', 'max_volume', 'cost_per_km'
    dist_matrix: 2D list of floats (square matrix)
    algorithm: 'ga' or 'vns'
    params: dict with algorithm-specific parameters
    Returns dict with solution details.
    """
    if not customers or not vehicles or not dist_matrix:
        raise ValueError("Missing customers, vehicles, or distance matrix")

    if len(dist_matrix) != len(customers) + 1 or any(len(row) != len(dist_matrix) for row in dist_matrix):
        raise ValueError("Distance matrix must be square and have size (n_customers + 1)")

    if params is None:
        params = {}

    if algorithm == 'ga':
        pop_size = params.get('population_size', 100)
        generations = params.get('generations', 100)
        cxpb = params.get('cxpb', 0.7)
        mutpb = params.get('mutpb', 0.2)
        start = time.time()
        best_solution, total_distance, total_cost, route_details = genetic_algorithm(
            customers, vehicles, dist_matrix, pop_size, generations, cxpb, mutpb
        )
        runtime = time.time() - start
    elif algorithm == 'vns':
        max_iter = params.get('max_iterations', 100)
        k_max = params.get('k_max', 5)
        start = time.time()
        best_solution, total_distance, total_cost, route_details = variable_neighbourhood_search(
            customers, vehicles, dist_matrix, max_iter, k_max
        )
        runtime = time.time() - start
    else:
        raise ValueError("Algorithm must be 'ga' or 'vns'")

    response = {
        'total_distance': total_distance,
        'total_cost': total_cost,
        'routes': route_details,
        'runtime': runtime,
        'algorithm': algorithm,
        'params': params,
        'customers': customers,
        'vehicles': vehicles
    }
    return response

# ---------- VRP Solve API Endpoint ----------
@app.route('/api/vrp/solve', methods=['POST'])
def solve_vrp_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON payload'}), 400

        customers = data.get('customers')
        vehicles = data.get('vehicles')
        dist_matrix = data.get('distance_matrix')
        algorithm = data.get('algorithm', 'ga')
        params = data.get('params', {})

        if not customers or not vehicles or not dist_matrix:
            return jsonify({'error': 'Missing customers, vehicles, or distance_matrix'}), 400

        result = solve_vrp(customers, vehicles, dist_matrix, algorithm, params)

        # Convert to serializable types
        for rd in result['routes']:
            rd['distance'] = float(rd['distance'])
            rd['cost'] = float(rd['cost'])
            rd['weight'] = float(rd['weight'])
            rd['volume'] = float(rd['volume'])
            rd['route'] = [int(x) for x in rd['route']]

        result['total_distance'] = float(result['total_distance'])
        result['total_cost'] = float(result['total_cost'])
        result['runtime'] = float(result['runtime'])

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================
#     NEW: INVENTORY CLASSIFICATION (ABC/XYZ/SS/RoP/EOQ)
#          backed by solvers/classifier.py + frontend/stock.html
# ============================================================

def _save_upload_temp(file_storage):
    """Save an uploaded FileStorage to a temp file and return its path."""
    suffix = os.path.splitext(file_storage.filename or 'upload.csv')[1] or '.csv'
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    file_storage.save(tmp_path)
    return tmp_path


def _read_csv_flexible(source):
    """
    Read a CSV into a DataFrame from either a file path (str) or an
    uploaded FileStorage object. Column names are stripped of whitespace.
    """
    if hasattr(source, 'read'):          # FileStorage / file-like
        df = pd.read_csv(source)
    else:                                # path on disk
        df = pd.read_csv(source)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _csv_product_dates(df):
    """Group Date values per Product_ID for frequency detection."""
    if 'Date' not in df.columns or 'Product_ID' not in df.columns:
        return {}
    product_dates = {}
    for pid, date in zip(df['Product_ID'], df['Date']):
        product_dates.setdefault(str(pid), []).append(date)
    return product_dates


def _parse_products_from_csv(df):
    """
    Validate a CSV DataFrame and build ProductInput objects.
    Mirrors solvers/classifier.load_csv but non-interactive: invalid rows are
    collected as warnings instead of prompting.
    Returns (products, validation_errors).
    """
    required_cols = [
        "Product_ID", "Product_Name", "Demand",
        "Unit_Cost", "Lead_Time_Days", "Ordering_Cost", "Holding_Cost_Rate"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    products = {}
    errors = []

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

        if pid not in products:
            # First occurrence of this product — parse static fields
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
                errors.append(f"Row {row_num} (Product {pid}): Invalid Holding_Cost_Rate '{row['Holding_Cost_Rate']}' - {e}")
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

    return list(products.values()), errors


def _parse_products_from_json(raw_products):
    """Build ProductInput objects from the manual-entry JSON payload."""
    products = []
    errors = []

    for i, p in enumerate(raw_products):
        label = f"Product {i + 1}"
        pid = str(p.get("product_id", "")).strip()
        if not pid:
            errors.append(f"{label}: Missing Product_ID")
            continue

        try:
            demand_history = parse_demand_history(str(p.get("demand_history", "")))
            if not demand_history:
                raise ValueError("At least one demand value is required.")
            if any(d < 0 for d in demand_history):
                raise ValueError("Demand values cannot be negative.")
        except ValueError as e:
            errors.append(f"{label} ({pid}): {e}")
            continue

        try:
            unit_cost = float(p.get("unit_cost", 0))
            if unit_cost < 0:
                raise ValueError("negative")
        except (ValueError, TypeError):
            errors.append(f"{label} ({pid}): Invalid Unit_Cost '{p.get('unit_cost')}'")
            continue

        try:
            lead_time = float(p.get("lead_time_days", 0))
            if lead_time < 0:
                raise ValueError("negative")
        except (ValueError, TypeError):
            errors.append(f"{label} ({pid}): Invalid Lead_Time_Days '{p.get('lead_time_days')}'")
            continue

        try:
            ordering_cost = float(p.get("ordering_cost", 0))
            if ordering_cost <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError):
            errors.append(f"{label} ({pid}): Invalid Ordering_Cost '{p.get('ordering_cost')}' (must be > 0)")
            continue

        try:
            holding_rate = normalize_percentage(p.get("holding_cost_rate", 0.25))
            if holding_rate <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError) as e:
            errors.append(f"{label} ({pid}): Invalid Holding_Cost_Rate '{p.get('holding_cost_rate')}' - {e}")
            continue

        products.append(ProductInput(
            product_id=pid,
            product_name=str(p.get("product_name") or pid).strip(),
            demand_history=demand_history,
            unit_cost=unit_cost,
            lead_time_days=lead_time,
            ordering_cost=ordering_cost,
            holding_cost_rate=holding_rate,
        ))

    return products, errors


def _resolve_settings(raw):
    """
    Normalize the settings dict sent by stock.html into the arguments
    expected by classifier.run_calculations.
    Returns (demand_period, service_level, z_value, abc_thresholds, xyz_thresholds).
    """
    demand_period = (raw.get("demand_period") or "weekly").lower().strip()
    if demand_period not in ("daily", "weekly", "monthly", "yearly"):
        raise ValueError(f"Invalid demand period: '{demand_period}'")

    # Service level / Z value
    sl_raw = str(raw.get("service_level", "95.0")).strip()
    if sl_raw.lower() == "custom":
        z_value = float(raw.get("custom_z") or 0)
        if z_value <= 0:
            raise ValueError("Custom Z value must be positive.")
        service_level = 0.0  # 0.0 indicates custom
    else:
        try:
            service_level = float(sl_raw)
        except ValueError:
            raise ValueError(f"Invalid service level: '{sl_raw}'")
        if service_level in SERVICE_LEVEL_Z_MAP:
            z_value = SERVICE_LEVEL_Z_MAP[service_level]
        else:
            # Snap to the nearest defined service level
            nearest = min(SERVICE_LEVEL_Z_MAP,
                          key=lambda k: abs(k - service_level))
            service_level, z_value = nearest, SERVICE_LEVEL_Z_MAP[nearest]

    # ABC thresholds
    def _flag(value, default=False):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("true", "1", "yes", "on", "")

    if _flag(raw.get("use_default_abc"), True):
        abc_thresholds = dict(DEFAULT_ABC_THRESHOLDS)
    else:
        a = float(raw.get("abc_a", 0.80))
        b = float(raw.get("abc_b", 0.95))
        if not (0 < a < b <= 1.0):
            raise ValueError("ABC thresholds must satisfy 0 < A < B <= 1.0")
        abc_thresholds = {"A": a, "B": b, "C": 1.0}

    # XYZ thresholds
    if _flag(raw.get("use_default_xyz"), True):
        xyz_thresholds = dict(DEFAULT_XYZ_THRESHOLDS)
    else:
        x = float(raw.get("xyz_x", 0.50))
        y = float(raw.get("xyz_y", 1.00))
        if not (0 <= x < y):
            raise ValueError("XYZ thresholds must satisfy 0 <= X < Y")
        xyz_thresholds = {"X": x, "Y": y, "Z": float("inf")}

    return demand_period, service_level, z_value, abc_thresholds, xyz_thresholds


def _product_result_to_row(r):
    """Convert a ProductResult into a JSON-safe dict for stock.html."""
    return {
        "product_id": r.product_id,
        "product_name": r.product_name,
        "mean_demand": r.mean_demand,
        "demand_stddev": r.demand_stddev,
        "annual_demand": r.annual_demand,
        "unit_cost": r.unit_cost,
        "annual_consumption_value": r.annual_consumption_value,
        "cumulative_acv_percentage": r.cumulative_acv_percentage,
        "abc_class": r.abc_class,
        "cv": r.cv if math.isfinite(r.cv) else None,
        "xyz_class": r.xyz_class,
        "combined_class": r.combined_class,
        "lead_time_days": r.lead_time_days,
        "lead_time_periods": r.lead_time_periods,
        "service_level": r.service_level,
        "z_value": r.z_value,
        "safety_stock": r.safety_stock,
        "lead_time_demand": r.lead_time_demand,
        "reorder_point": r.reorder_point,
        "ordering_cost": r.ordering_cost,
        "holding_cost_rate": r.holding_cost_rate,
        "annual_holding_cost_per_unit": r.annual_holding_cost_per_unit,
        "eoq": r.eoq,
    }


def _build_response(results, validation_errors):
    """Build the {summary, rows, validation_errors} payload for stock.html."""
    n_products = len(results)
    total_acv = sum(r.annual_consumption_value for r in results)

    abc_counts = {"A": 0, "B": 0, "C": 0}
    xyz_counts = {"X": 0, "Y": 0, "Z": 0}
    for r in results:
        abc_counts[r.abc_class] = abc_counts.get(r.abc_class, 0) + 1
        xyz_counts[r.xyz_class] = xyz_counts.get(r.xyz_class, 0) + 1

    return {
        "summary": {
            "n_products": n_products,
            "total_acv": total_acv,
            "abc_counts": abc_counts,
            "xyz_counts": xyz_counts,
        },
        "rows": [_product_result_to_row(r) for r in results],
        "validation_errors": validation_errors,
    }


@app.route('/api/detect-frequency', methods=['POST'])
def detect_frequency():
    """
    Detect demand frequency from an uploaded CSV's Date column.
    Used by stock.html right after the user drops a file.
    Returns: {"detected": "weekly"} or {"detected": null, "reason": "..."}
    """
    if 'file' not in request.files:
        return jsonify({'detected': None, 'reason': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'detected': None, 'reason': 'Empty filename'}), 400

    tmp_path = None
    try:
        tmp_path = _save_upload_temp(file)
        df = _read_csv_flexible(tmp_path)
        product_dates = _csv_product_dates(df)
        if not product_dates:
            return jsonify({'detected': None,
                            'reason': 'Missing Date or Product_ID column'})
        detected = detect_period_from_products(product_dates)
        if detected:
            return jsonify({'detected': detected})
        return jsonify({'detected': None,
                        'reason': 'Could not infer from dates (insufficient rows)'})
    except Exception as e:
        return jsonify({'detected': None, 'reason': str(e)}), 400
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route('/api/calculate', methods=['POST'])
def calculate_inventory():
    """
    Run the full ABC/XYZ/SS/RoP/EOQ pipeline.

    Accepts either:
      - multipart/form-data with a CSV file in 'file' + form settings, or
      - application/json with {"products": [...], "settings": {...}}.
    """
    validation_errors = []
    try:
        # --------------------------------------------------------------
        # 1. Collect settings (form fields or JSON body)
        # --------------------------------------------------------------
        if request.content_type and 'application/json' in request.content_type:
            payload = request.get_json(force=True) or {}
            raw_products = payload.get('products') or []
            settings = payload.get('settings') or {}
            products, errors = _parse_products_from_json(raw_products)
            validation_errors.extend(errors)
        else:
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'Empty filename'}), 400
            settings = request.form.to_dict()
            tmp_path = _save_upload_temp(file)
            try:
                df = _read_csv_flexible(tmp_path)
                products, errors = _parse_products_from_csv(df)
                validation_errors.extend(errors)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if not products:
            return jsonify({
                'error': 'No valid products could be loaded.',
                'validation_errors': validation_errors,
            }), 400

        # --------------------------------------------------------------
        # 2. Resolve settings and run the classifier pipeline
        # --------------------------------------------------------------
        (demand_period, service_level, z_value,
         abc_thresholds, xyz_thresholds) = _resolve_settings(settings)

        results = run_calculations(
            products,
            demand_period,
            service_level,
            z_value,
            abc_thresholds,
            xyz_thresholds,
        )

        # --------------------------------------------------------------
        # 3. Build response
        # --------------------------------------------------------------
        response = _build_response(results, validation_errors)
        response['settings'] = {
            'demand_period': demand_period,
            'service_level': service_level,
            'z_value': z_value,
        }
        return jsonify(response)

    except ValueError as e:
        return jsonify({'error': str(e),
                        'validation_errors': validation_errors}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================
#           NEW: CATBOOST DEMAND FORECASTING ENDPOINTS
# ============================================================

@app.route('/api/forecast/train', methods=['POST'])
def forecast_train():
    """
    Train a CatBoost demand forecasting model from an uploaded file.
    Expects a multipart/form-data with a file field 'file' (CSV or Excel).
    Optional JSON form-data fields: test_size, model_params (JSON string).
    Returns training metrics and data summary.
    """
    global forecaster, forecaster_metrics, forecaster_summary

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Read file content
    try:
        # Determine file extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in ['.csv']:
            import pandas as pd
            df = pd.read_csv(file.stream)
        elif ext in ['.xlsx', '.xls', '.xlsm']:
            import pandas as pd
            df = pd.read_excel(file.stream)
        else:
            return jsonify({'error': 'Unsupported file format. Use .csv, .xlsx, .xls, or .xlsm'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to read file: {str(e)}'}), 400

    # Optional parameters
    test_size = float(request.form.get('test_size', 0.2))
    model_params_str = request.form.get('model_params', '{}')
    try:
        model_params = json.loads(model_params_str)
    except json.JSONDecodeError:
        model_params = {}

    # Create and train forecaster
    try:
        forecaster = DemandForecaster(model_params=model_params, verbose=False)
        forecaster.set_data(df)
        summary = forecaster.get_data_summary()
        metrics = forecaster.train(test_size=test_size)
        # Also compute feature importance
        importance = forecaster.get_feature_importance()
        forecaster_metrics = metrics
        forecaster_summary = summary
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

    response = {
        'metrics': metrics,
        'summary': summary,
        'feature_importance': importance,
        'message': 'Model trained successfully'
    }
    return jsonify(response), 200

@app.route('/api/forecast/predict', methods=['POST'])
def forecast_predict():
    """
    Generate demand forecasts using the trained model.
    Expects JSON with:
        - product_id (string, optional, defaults to first product)
        - days (integer)
        - future_promotion (dict or integer, optional)
        - future_discount (dict or float, optional)
        - future_holiday (dict or integer, optional)
        - future_opening_stock (dict or float, optional)
    Returns forecast list.
    """
    global forecaster
    if forecaster is None:
        return jsonify({'error': 'No trained model. Call /api/forecast/train first.'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON payload'}), 400

    product_id = data.get('product_id')
    days = data.get('days', 7)
    future_promotion = data.get('future_promotion', 0)
    future_discount = data.get('future_discount', 0.0)
    future_holiday = data.get('future_holiday', 0)
    future_opening_stock = data.get('future_opening_stock')

    try:
        forecast_result = forecaster.forecast(
            days=days,
            product_id=product_id,
            future_promotion=future_promotion,
            future_discount=future_discount,
            future_holiday=future_holiday,
            future_opening_stock=future_opening_stock
        )
        # forecast_result is dict: {product_id: list of forecast dicts}
        return jsonify(forecast_result), 200
    except Exception as e:
        return jsonify({'error': f'Forecast failed: {str(e)}'}), 500

@app.route('/api/forecast/status', methods=['GET'])
def forecast_status():
    """Return current model status and metrics."""
    global forecaster, forecaster_metrics, forecaster_summary
    if forecaster is None:
        return jsonify({'trained': False, 'message': 'No model trained'}), 200
    return jsonify({
        'trained': True,
        'metrics': forecaster_metrics,
        'summary': forecaster_summary
    }), 200

@app.route('/api/forecast/feature_importance', methods=['GET'])
def forecast_feature_importance():
    """Return feature importance of the trained model."""
    global forecaster
    if forecaster is None:
        return jsonify({'error': 'No trained model'}), 400
    try:
        importance = forecaster.get_feature_importance()
        return jsonify({'feature_importance': importance}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast/event_impact', methods=['GET'])
def forecast_event_impact():
    """Return impact of promotions on demand (using promotion column)."""
    global forecaster
    if forecaster is None:
        return jsonify({'error': 'No trained model'}), 400
    try:
        impact = forecaster.get_event_impact()
        return jsonify({'event_impact': impact}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast', methods=['POST'])
def forecast_combined():
    """One-shot endpoint: train + forecast. Matches forecast.html payload."""
    global forecaster, forecaster_metrics, forecaster_summary

    try:
        payload = request.get_json(force=True) or {}
        raw = payload.get('data')
        if not raw:
            return jsonify({'error': 'Missing "data" field'}), 400

        test_size      = float(payload.get('test_size', 0.2))
        forecast_days  = int(payload.get('forecast_days', 7))
        future_events  = payload.get('future_events', {}) or {}
        opening_stock  = payload.get('opening_stock', 300)

        df = pd.DataFrame(raw)

        # --- Train ---
        forecaster = DemandForecaster(verbose=False)
        forecaster.set_data(df)
        forecaster_summary = forecaster.get_data_summary()
        forecaster_metrics = forecaster.train(test_size=test_size)

        # --- Test-set predictions for the chart ---
        test_preds = forecaster.get_train_test_predictions()

        # --- Feature importance ---
        importance = forecaster.get_feature_importance()

        # --- Forecast each product we trained on ---
        forecast_all = {}
        for pid in forecaster.last_row['product_id'].unique():
            res = forecaster.forecast(
                days=forecast_days,
                product_id=str(pid),
                future_opening_stock=opening_stock,
            )
            forecast_all.update(res)

        # Flatten to a single list the table can iterate
        flat = []
        for pid, rows in forecast_all.items():
            for r in rows:
                flat.append({
                    'date':             r['date'],
                    'day_name':         r['day_name'],
                    'product_id':       r.get('product_id', pid),
                    'predicted_demand': r['predicted_demand'],
                    'event':            future_events.get(r['date'], ''),
                    'opening_stock':    r.get('opening_stock', ''),
                    'replenishment':    '',   # not modelled by CatBoost solver
                })

        return jsonify({
            'metrics':            forecaster_metrics,
            'summary':            forecaster_summary,
            'test_predictions':   test_preds,
            'forecast':           flat,
            'feature_importance': importance,
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================
#   NEW: SHARED DEMAND — feed the Forecast page from the DB
# ============================================================
@app.route('/api/forecast/data', methods=['GET'])
def api_forecast_data():
    """
    Return every product's demand history in the flat, column-oriented
    format that DemandForecaster / /api/forecast expects.

    Query params:
        product_id   (optional)  restrict to a single product id

    Response:
        {
          "data": [
             {"Product_ID": "...", "Product_Name": "...", "Date": "YYYY-MM-DD",
              "Demand_Units": 123, "Opening_Stock_Units": 0,
              "Replenishment_Units": 0},
             ...
          ],
          "products": [{"id": 1, "name": "...", "sku": "..."}, ...]
        }
    """
    try:
        pid_filter = request.args.get('product_id')
        products = get_products_with_demand()

        rows = []
        for p in products:
            if pid_filter and str(p['id']) != str(pid_filter):
                continue
            for rec in p.get('demand_history', []):
                rows.append({
                    'Product_ID':           str(p['id']),
                    'Product_Name':         p['name'],
                    'Date':                 rec.get('date'),           # may be None
                    'Demand_Units':         rec.get('demand') or 0,
                    'Opening_Stock_Units':  0,     # not stored in the DB yet
                    'Replenishment_Units':  0,     # not stored in the DB yet
                })

        return jsonify({
            'data': rows,
            'products': [
                {'id': p['id'], 'name': p['name'], 'sku': p.get('sku', '')}
                for p in products
            ],
        }), 200

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================
#          SHARED DATA — Workstations (from FJSP)
# ============================================================

@app.route('/api/workstations', methods=['GET'])
def api_workstations_list():
    """Return the workstations currently stored in the shared DB."""
    try:
        return jsonify({'workstations': get_workstations()}), 200
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/workstations', methods=['POST'])
def api_workstations_save():
    """
    Replace the stored workstation set.
    Body: {"workstations": [{"name": "Cutting", "machines": 2}, ...]}
    """
    try:
        data = request.get_json(force=True) or {}
        items = data.get('workstations')
        if not isinstance(items, list):
            return jsonify({'error': 'Expected "workstations" array'}), 400
        saved = replace_workstations(items)
        return jsonify({'workstations': saved}), 200
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/workstations', methods=['DELETE'])
def api_workstations_clear():
    """Remove every workstation from the shared DB."""
    try:
        clear_workstations()
        return jsonify({'ok': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/workstations/<int:ws_id>', methods=['GET'])
def api_workstation_get(ws_id):
    ws = get_workstation(ws_id)
    if not ws:
        return jsonify({'error': 'Workstation not found'}), 404
    return jsonify({'workstation': ws}), 200


@app.route('/api/workstations/create', methods=['POST'])
def api_workstation_create():
    data = request.get_json(force=True) or {}
    try:
        ws = create_workstation(data.get('name'), data.get('machines', 1))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    return jsonify({'workstation': ws}), 201


@app.route('/api/workstations/<int:ws_id>', methods=['PUT'])
def api_workstation_update(ws_id):
    data = request.get_json(force=True) or {}
    try:
        ws = update_workstation(ws_id, data.get('name'), data.get('machines', []))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    if not ws:
        return jsonify({'error': 'Workstation not found'}), 404
    return jsonify({'workstation': ws}), 200


@app.route('/api/workstations/<int:ws_id>', methods=['DELETE'])
def api_workstation_delete(ws_id):
    delete_workstation(ws_id)
    return jsonify({'ok': True}), 200

# ============================================================
#     SHARED DATA — Products (Stock tab -> FJSP job names)
# ============================================================

@app.route('/api/products', methods=['GET'])
def api_products_list():
    try:
        return jsonify({'products': get_products()}), 200
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/products', methods=['POST'])
def api_products_create():
    data = request.get_json(force=True) or {}
    try:
        p = create_product(
            name=data.get('name'),
            sku=data.get('sku', ''),
            qty=data.get('qty', 0),
            unit=data.get('unit', 'pcs'),
            price=data.get('price', 0),
            restock_day=data.get('restock_day'),
            image=data.get('image'),
            unit_cost=data.get('unit_cost', 0),
            lead_time_days=data.get('lead_time_days', 0),
            ordering_cost=data.get('ordering_cost', 0),
            holding_cost_rate=data.get('holding_cost_rate', 0.25),
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    return jsonify({'product': p}), 201


@app.route('/api/products/<int:pid>', methods=['GET'])
def api_products_get(pid):
    p = get_product(pid)
    if not p:
        return jsonify({'error': 'Product not found'}), 404
    return jsonify({'product': p}), 200


@app.route('/api/products/<int:pid>', methods=['PUT'])
def api_products_update(pid):
    data = request.get_json(force=True) or {}
    try:
        p = update_product(
            pid,
            name=data.get('name'),
            sku=data.get('sku', ''),
            qty=data.get('qty', 0),
            unit=data.get('unit', 'pcs'),
            price=data.get('price', 0),
            restock_day=data.get('restock_day'),
            image=data.get('image'),
            unit_cost=data.get('unit_cost', 0),
            lead_time_days=data.get('lead_time_days', 0),
            ordering_cost=data.get('ordering_cost', 0),
            holding_cost_rate=data.get('holding_cost_rate', 0.25),
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    if not p:
        return jsonify({'error': 'Product not found'}), 404
    return jsonify({'product': p}), 200


@app.route('/api/products/<int:pid>', methods=['DELETE'])
def api_products_delete(pid):
    try:
        delete_product(pid)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify({'ok': True}), 200

# ============================================================
#   PRODUCT DEMAND HISTORY (Stock page ↔ DB)
# ============================================================
@app.route('/api/products/with-demand', methods=['GET'])
def api_products_with_demand():
    """All products, each with its full demand history."""
    try:
        return jsonify({'products': get_products_with_demand()}), 200
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/products/<int:pid>/demand', methods=['GET'])
def api_product_demand_get(pid):
    if not get_product(pid):
        return jsonify({'error': 'Product not found'}), 404
    return jsonify({'demand': get_product_demand(pid)}), 200


@app.route('/api/products/<int:pid>/demand', methods=['POST'])
def api_product_demand_save(pid):
    if not get_product(pid):
        return jsonify({'error': 'Product not found'}), 404
    data = request.get_json(force=True) or {}
    try:
        n = save_product_demand(pid, data.get('records', []),
                                replace=bool(data.get('replace', True)))
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'saved': n, 'demand': get_product_demand(pid)}), 200


def _parse_csv_for_db(df):
    """
    Parse a raw CSV DataFrame into a list of product dicts ready to be
    upserted. Groups rows by Product_ID and keeps the Date column as the
    record date (if present).
    """
    required = ["Product_ID", "Product_Name", "Demand",
                "Unit_Cost", "Lead_Time_Days", "Ordering_Cost", "Holding_Cost_Rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    has_date = "Date" in df.columns
    out: dict[str, dict] = {}

    for _, row in df.iterrows():
        pid = str(row.get("Product_ID", "")).strip()
        if not pid:
            continue
        try:
            demand = float(row["Demand"])
        except (ValueError, TypeError):
            continue
        if demand < 0:
            continue

        if pid not in out:
            try:
                out[pid] = {
                    "product_id":         pid,
                    "name":               str(row.get("Product_Name", pid)).strip() or pid,
                    "unit_cost":          float(row["Unit_Cost"]),
                    "lead_time_days":     float(row["Lead_Time_Days"]),
                    "ordering_cost":      float(row["Ordering_Cost"]),
                    "holding_cost_rate":  normalize_percentage(row["Holding_Cost_Rate"]),
                    "records":            [],
                }
            except (ValueError, TypeError):
                continue

        date_val = None
        if has_date:
            v = row.get("Date")
            if not pd.isna(v):
                date_val = str(v)[:10]

        out[pid]["records"].append({"date": date_val, "demand": demand})

    return list(out.values())


@app.route('/api/stock/import-csv', methods=['POST'])
def api_stock_import_csv():
    """
    Import an uploaded inventory CSV into the DB:
      - upserts a product per unique Product_ID (matched by SKU)
      - replaces its demand history
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    tmp_path = None
    try:
        tmp_path = _save_upload_temp(file)
        df = _read_csv_flexible(tmp_path)
        products = _parse_csv_for_db(df)
        if not products:
            return jsonify({'error': 'No valid products found in CSV.'}), 400

        saved = []
        for p in products:
            rec = upsert_product_with_demand(
                product_id         = p["product_id"],
                name               = p["name"],
                unit_cost          = p["unit_cost"],
                lead_time_days     = p["lead_time_days"],
                ordering_cost      = p["ordering_cost"],
                holding_cost_rate  = p["holding_cost_rate"],
                records            = p["records"],
            )
            saved.append({
                "id":             rec["id"],
                "product_id":     p["product_id"],
                "name":           rec["name"],
                "records_saved":  len(rec.get("demand_history", [])),
            })

        return jsonify({"imported": len(saved), "products": saved}), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================
#   SHARED DATA — Clients (declared on Home, consumed by VRP)
# ============================================================
@app.route('/api/clients', methods=['GET'])
def api_clients_list():
    try:
        return jsonify({'clients': get_clients()}), 200
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/clients', methods=['POST'])
def api_clients_create():
    data = request.get_json(force=True) or {}
    try:
        c = create_client(
            data.get('name'),
            contact=data.get('contact', ''),
            email=data.get('email', ''),
            phone=data.get('phone', ''),
            address=data.get('address', ''),
            image=data.get('image'),
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    return jsonify({'client': c}), 201


@app.route('/api/clients/<int:cid>', methods=['GET'])
def api_clients_get(cid):
    c = get_client(cid)
    if not c:
        return jsonify({'error': 'Client not found'}), 404
    return jsonify({'client': c}), 200


@app.route('/api/clients/<int:cid>', methods=['PUT'])
def api_clients_update(cid):
    data = request.get_json(force=True) or {}
    try:
        c = update_client(
            cid, data.get('name'),
            contact=data.get('contact', ''),
            email=data.get('email', ''),
            phone=data.get('phone', ''),
            address=data.get('address', ''),
            image=data.get('image'),
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    if not c:
        return jsonify({'error': 'Client not found'}), 404
    return jsonify({'client': c}), 200


@app.route('/api/clients/<int:cid>', methods=['DELETE'])
def api_clients_delete(cid):
    try:
        delete_client(cid)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify({'ok': True}), 200


# ============================================================
#   SHARED DATA — Trucks (managed on Home → Machines tab)
# ============================================================
@app.route('/api/trucks', methods=['GET'])
def api_trucks_list():
    try:
        return jsonify({'trucks': get_trucks()}), 200
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/trucks', methods=['POST'])
def api_trucks_create():
    data = request.get_json(force=True) or {}
    try:
        t = create_truck(
            data.get('name'),
            cost_per_km=data.get('cost_per_km', 1.0),
            capacities=data.get('capacities') or {},
            notes=data.get('notes', ''),
            image=data.get('image'),
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    return jsonify({'truck': t}), 201


@app.route('/api/trucks/<int:tid>', methods=['GET'])
def api_trucks_get(tid):
    t = get_truck(tid)
    if not t:
        return jsonify({'error': 'Truck not found'}), 404
    return jsonify({'truck': t}), 200


@app.route('/api/trucks/<int:tid>', methods=['PUT'])
def api_trucks_update(tid):
    data = request.get_json(force=True) or {}
    try:
        t = update_truck(
            tid, data.get('name'),
            cost_per_km=data.get('cost_per_km', 1.0),
            capacities=data.get('capacities') or {},
            notes=data.get('notes', ''),
            image=data.get('image'),
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    if not t:
        return jsonify({'error': 'Truck not found'}), 404
    return jsonify({'truck': t}), 200


@app.route('/api/trucks/<int:tid>', methods=['DELETE'])
def api_trucks_delete(tid):
    try:
        delete_truck(tid)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify({'ok': True}), 200


# ============================================================
#   SHARED DATA — Suppliers
# ============================================================
@app.route('/api/suppliers', methods=['GET'])
def api_suppliers_list():
    try:
        return jsonify({'suppliers': get_suppliers()}), 200
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/suppliers', methods=['POST'])
def api_suppliers_create():
    data = request.get_json(force=True) or {}
    try:
        s = create_supplier(
            data.get('name'),
            contact=data.get('contact', ''),
            email=data.get('email', ''),
            phone=data.get('phone', ''),
            address=data.get('address', ''),
            image=data.get('image'),
            product_ids=data.get('product_ids') or [],
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    return jsonify({'supplier': s}), 201


@app.route('/api/suppliers/<int:sid>', methods=['GET'])
def api_suppliers_get(sid):
    s = get_supplier(sid)
    if not s:
        return jsonify({'error': 'Supplier not found'}), 404
    return jsonify({'supplier': s}), 200


@app.route('/api/suppliers/<int:sid>', methods=['PUT'])
def api_suppliers_update(sid):
    data = request.get_json(force=True) or {}
    try:
        s = update_supplier(
            sid, data.get('name'),
            contact=data.get('contact', ''),
            email=data.get('email', ''),
            phone=data.get('phone', ''),
            address=data.get('address', ''),
            image=data.get('image'),
            product_ids=data.get('product_ids') or [],
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    if not s:
        return jsonify({'error': 'Supplier not found'}), 404
    return jsonify({'supplier': s}), 200


@app.route('/api/suppliers/<int:sid>', methods=['DELETE'])
def api_suppliers_delete(sid):
    try:
        delete_supplier(sid)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify({'ok': True}), 200


# ============================================================
#   SHARED DATA — Employees (+ shift log)
# ============================================================
@app.route('/api/employees', methods=['GET'])
def api_employees_list():
    try:
        return jsonify({'employees': get_employees()}), 200
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/employees', methods=['POST'])
def api_employees_create():
    data = request.get_json(force=True) or {}
    try:
        e = create_employee(
            data.get('name'),
            age=data.get('age'),
            sex=data.get('sex', ''),
            birthday=data.get('birthday'),
            email=data.get('email', ''),
            phone=data.get('phone', ''),
            address=data.get('address', ''),
            image=data.get('image'),
            pay_type=data.get('pay_type', 'hourly'),
            rate=data.get('rate', 0),
            pay_day=data.get('pay_day'),
        )
    except ValueError as err:
        return jsonify({'error': str(err)}), 400
    except Exception as err:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(err)}), 500
    return jsonify({'employee': e}), 201


@app.route('/api/employees/<int:eid>', methods=['GET'])
def api_employees_get(eid):
    e = get_employee(eid)
    if not e:
        return jsonify({'error': 'Employee not found'}), 404
    return jsonify({'employee': e}), 200


@app.route('/api/employees/<int:eid>', methods=['PUT'])
def api_employees_update(eid):
    data = request.get_json(force=True) or {}
    try:
        e = update_employee(
            eid, data.get('name'),
            age=data.get('age'),
            sex=data.get('sex', ''),
            birthday=data.get('birthday'),
            email=data.get('email', ''),
            phone=data.get('phone', ''),
            address=data.get('address', ''),
            image=data.get('image'),
            pay_type=data.get('pay_type', 'hourly'),
            rate=data.get('rate', 0),
            pay_day=data.get('pay_day'),
            shifts=data.get('shifts'),   # None => don't touch shifts
        )
    except ValueError as err:
        return jsonify({'error': str(err)}), 400
    except Exception as err:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(err)}), 500
    if not e:
        return jsonify({'error': 'Employee not found'}), 404
    return jsonify({'employee': e}), 200


@app.route('/api/employees/<int:eid>', methods=['DELETE'])
def api_employees_delete(eid):
    try:
        delete_employee(eid)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify({'ok': True}), 200


# ============================================================
#   SHARED DATA — Client distances (auto-built matrix)
# ============================================================
@app.route('/api/distances', methods=['GET'])
def api_distances_get():
    try:
        return jsonify(get_distance_matrix()), 200
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/distances', methods=['POST'])
def api_distances_set():
    data = request.get_json(force=True) or {}
    try:
        if 'matrix' in data and 'client_ids' in data:
            set_distance_matrix(data['matrix'], data['client_ids'])
        else:
            f = int(data.get('from_id', 0))
            t = int(data.get('to_id', 0))
            d = data.get('distance', None)
            set_distance(f, t, None if d in (None, '', 'None') else float(d))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    return jsonify({'ok': True}), 200

# ---------- Serve static files ----------
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)