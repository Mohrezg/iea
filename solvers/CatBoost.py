"""
CatBoost Demand Forecasting — Backend Solver (Multi‑Product, Flexible Columns)
=============================================================================
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


class DemandForecaster:
    # Updated required columns – we'll map your column names to these internally
    REQUIRED_COLUMNS = ["Date", "Demand_Units", "Opening_Stock_Units"]

    # Feature set – now includes promotion, discount, holiday, and product_id as categorical
    FEATURE_COLS = [
        "day_of_week", "day_of_month", "month", "week_of_year",
        "is_weekend", "is_month_start", "is_month_end",
        "demand_lag_1", "demand_lag_2", "demand_lag_3", "demand_lag_7",
        "demand_roll_mean_3", "demand_roll_mean_7",
        "demand_roll_std_3", "demand_roll_std_7",
        "Opening_Stock_Units",
        "stock_to_demand_ratio",
        "promotion", "discount_percent", "is_holiday",
    ]

    CATEGORICAL_FEATURES = ["product_id"]   # product_id as categorical

    def __init__(self, model_params: Optional[Dict[str, Any]] = None, verbose: bool = False):
        self.verbose = verbose
        self.model: Optional[CatBoostRegressor] = None

        self.df: Optional[pd.DataFrame] = None
        self.df_features: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.last_row: Optional[pd.Series] = None

        self.metrics: Dict[str, float] = {}
        self.y_test: Optional[pd.Series] = None
        self.y_pred: Optional[np.ndarray] = None

        defaults = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "RMSE",
            "eval_metric": "MAE",
            "random_seed": 42,
            "early_stopping_rounds": 50,
        }
        self.model_params = {**defaults, **(model_params or {})}
        self.model_params["verbose"] = 50 if verbose else False

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # --------------------------------------------------------------------- #
    #  Data loading & validation
    # --------------------------------------------------------------------- #
    def load_data(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in (".xlsx", ".xls", ".xlsm"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported format '{ext}'. Use .csv, .xlsx, .xls or .xlsm")

        return self.set_data(df)

    def set_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ----- Column mapping for your dataset -----
        # Rename your columns to the internal names expected by the model
        rename_map = {}
        if "demand_quantity" in df.columns:
            rename_map["demand_quantity"] = "Demand_Units"
        if "opening_stock" in df.columns:
            rename_map["opening_stock"] = "Opening_Stock_Units"
        if "date" in df.columns:
            rename_map["date"] = "Date"
        # product_id is used as categorical, keep as is
        # promotion, discount_percent, is_holiday stay as they are
        df.rename(columns=rename_map, inplace=True)

        # Check required columns exist (after renaming)
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns (after renaming): {missing}")

        # Ensure product_id exists; if not, create a dummy one
        if "product_id" not in df.columns:
            df["product_id"] = "all_products"
            self.CATEGORICAL_FEATURES = []  # no categorical if single product

        # Harden: coerce numeric columns
        numeric_cols = ["Demand_Units", "Opening_Stock_Units", "promotion",
                        "discount_percent", "is_holiday"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Ensure Date is datetime
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if df["Date"].isna().any():
            raise ValueError("Some dates could not be parsed. Use YYYY-MM-DD format.")

        # Sort by product and date
        df = df.sort_values(["product_id", "Date"]).reset_index(drop=True)

        # For compatibility, we keep an 'Event' column as None (not used)
        if "Event" not in df.columns:
            df["Event"] = "None"

        self.df = df
        self._log(f"Loaded {len(df)} rows | {df['Date'].min().date()} → {df['Date'].max().date()} "
                  f"| Products: {df['product_id'].nunique()}")
        return self.df

    def get_data_summary(self) -> Dict[str, Any]:
        if self.df is None:
            raise ValueError("No data loaded.")
        return {
            "total_rows": len(self.df),
            "date_range": {
                "start": self.df["Date"].min().strftime("%Y-%m-%d"),
                "end": self.df["Date"].max().strftime("%Y-%m-%d"),
            },
            "products": sorted(self.df["product_id"].unique().tolist()),
            "avg_demand": round(float(self.df["Demand_Units"].mean()), 2),
            "avg_opening_stock": round(float(self.df["Opening_Stock_Units"].mean()), 2),
        }

    # --------------------------------------------------------------------- #
    #  Feature engineering (now group‑aware)
    # --------------------------------------------------------------------- #
    def _create_features(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input.copy()
        # Ensure sorted by date per product
        df = df.sort_values(["product_id", "Date"]).reset_index(drop=True)

        # Time features (no grouping needed)
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["day_of_month"] = df["Date"].dt.day
        df["month"] = df["Date"].dt.month
        df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_month_start"] = (df["day_of_month"] <= 5).astype(int)
        df["is_month_end"] = (df["day_of_month"] >= 25).astype(int)

        # Lag and rolling features per product
        # We'll use groupby to compute shifts within each product
        df["demand_lag_1"] = df.groupby("product_id")["Demand_Units"].shift(1)
        df["demand_lag_2"] = df.groupby("product_id")["Demand_Units"].shift(2)
        df["demand_lag_3"] = df.groupby("product_id")["Demand_Units"].shift(3)
        df["demand_lag_7"] = df.groupby("product_id")["Demand_Units"].shift(7)

        # Rolling means / std (using shift to avoid look‑ahead)
        df["demand_roll_mean_3"] = df.groupby("product_id")["Demand_Units"].shift(1).rolling(
            window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        df["demand_roll_mean_7"] = df.groupby("product_id")["Demand_Units"].shift(1).rolling(
            window=7, min_periods=1).mean().reset_index(level=0, drop=True)
        df["demand_roll_std_3"] = df.groupby("product_id")["Demand_Units"].shift(1).rolling(
            window=3, min_periods=1).std().reset_index(level=0, drop=True)
        df["demand_roll_std_7"] = df.groupby("product_id")["Demand_Units"].shift(1).rolling(
            window=7, min_periods=1).std().reset_index(level=0, drop=True)

        # Stock‑to‑demand ratio (using rolling mean of 7 days)
        df["stock_to_demand_ratio"] = df["Opening_Stock_Units"] / (df["demand_roll_mean_7"] + 1)

        # Drop rows with NaN lags (first few rows per product)
        df = df.dropna(
            subset=["demand_lag_1", "demand_lag_2", "demand_lag_3", "demand_lag_7"]
        ).reset_index(drop=True)

        return df

    def prepare_features(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() or set_data() first.")
        if len(self.df) < 8:
            raise ValueError(f"Need ≥8 rows per product, got {len(self.df)}")

        self.df_features = self._create_features(self.df)
        self.last_row = self.df_features.groupby("product_id").last().reset_index()
        self._log(f"Features ready: {len(self.df_features)} rows")
        return self.df_features

    # --------------------------------------------------------------------- #
    #  Training (now per‑product split kept chronological)
    # --------------------------------------------------------------------- #
    def train(self, test_size: float = 0.2) -> Dict[str, float]:
        if self.df_features is None:
            self.prepare_features()

        # We'll split by product? Better to split globally but keep time order.
        # For simplicity, we do a global chronological split.
        n = len(self.df_features)
        if n < 15:
            raise ValueError(f"Need ≥15 rows after feature engineering, got {n}")

        split_idx = int(n * (1 - test_size))
        self.train_df = self.df_features.iloc[:split_idx].copy()
        self.test_df = self.df_features.iloc[split_idx:].copy()

        self._log(f"Train: {len(self.train_df)} rows | Test: {len(self.test_df)} rows")

        X_train = self.train_df[self.FEATURE_COLS]
        y_train = self.train_df["Demand_Units"]
        X_test = self.test_df[self.FEATURE_COLS]
        y_test = self.test_df["Demand_Units"]

        # Ensure categorical columns exist and are included
        cat_features = [c for c in self.CATEGORICAL_FEATURES if c in X_train.columns]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        test_pool = Pool(X_test, y_test, cat_features=cat_features)

        self.model = CatBoostRegressor(**self.model_params)
        self.model.fit(train_pool, eval_set=test_pool)

        self.y_pred = self.model.predict(X_test)
        self.y_test = y_test

        mae = mean_absolute_error(y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, self.y_pred))
        r2 = r2_score(y_test, self.y_pred)
        mape = (mae / y_test.mean() * 100) if y_test.mean() != 0 else float("inf")

        self.metrics = {
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "r2": round(float(r2), 4),
            "mape": round(float(mape), 2),
        }

        self._log(
            f"Training complete — MAE:{mae:.2f} RMSE:{rmse:.2f} R²:{r2:.4f} MAPE:{mape:.2f}%"
        )
        return self.metrics

    # --------------------------------------------------------------------- #
    #  Post‑training insights (feature importance, etc.)
    # --------------------------------------------------------------------- #
    def get_feature_importance(self) -> List[Dict[str, Any]]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        imp = self.model.get_feature_importance()
        result = [
            {"feature": f, "importance": round(float(i), 2)}
            for f, i in zip(self.FEATURE_COLS, imp)
        ]
        return sorted(result, key=lambda x: x["importance"], reverse=True)

    def get_event_impact(self) -> List[Dict[str, Any]]:
        # This method originally used an 'Event' column; we'll adapt to use 'promotion' as event
        if self.model is None or self.df_features is None:
            raise ValueError("Model not trained.")
        preds = self.model.predict(self.df_features[self.FEATURE_COLS])
        analysis = self.df_features.copy()
        analysis["predicted_demand"] = preds

        # Group by promotion status (0/1) to measure impact
        stats = (
            analysis.groupby("promotion")
            .agg({"Demand_Units": ["mean", "count"], "predicted_demand": "mean"})
            .round(2)
        )
        stats.columns = ["actual_avg", "count", "predicted_avg"]
        stats = stats.reset_index()

        baseline_row = stats[stats["promotion"] == 0]
        baseline = (
            float(baseline_row["actual_avg"].values[0])
            if len(baseline_row) > 0
            else float(stats["actual_avg"].mean())
        )

        out: List[Dict[str, Any]] = []
        for _, row in stats.iterrows():
            impact = float(row["actual_avg"]) - baseline
            impact_pct = (impact / baseline * 100) if baseline != 0 else 0.0
            out.append(
                {
                    "promotion": int(row["promotion"]),
                    "count": int(row["count"]),
                    "actual_avg": round(float(row["actual_avg"]), 2),
                    "predicted_avg": round(float(row["predicted_avg"]), 2),
                    "impact_pct": round(impact_pct, 2),
                }
            )
        return sorted(out, key=lambda x: x["actual_avg"], reverse=True)

    def get_train_test_predictions(self) -> Dict[str, Any]:
        if self.model is None or self.test_df is None:
            raise ValueError("Model not trained.")
        return {
            "dates": self.test_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "products": self.test_df["product_id"].tolist(),
            "actual": self.y_test.tolist(),
            "predicted": self.y_pred.tolist(),
        }

    def predict_historical(self) -> pd.DataFrame:
        if self.model is None or self.df_features is None:
            raise ValueError("Model not trained.")
        df = self.df_features[["Date", "product_id", "Demand_Units", "promotion"]].copy()
        df["predicted_demand"] = self.model.predict(self.df_features[self.FEATURE_COLS])
        return df

    # --------------------------------------------------------------------- #
    #  Forecasting (now with promotion/discount/holiday inputs)
    # --------------------------------------------------------------------- #
    def forecast(
        self,
        days: int = 7,
        product_id: str = None,   # if None, forecast for all products? We'll require one product
        future_promotion: Optional[Union[int, Dict[str, int]]] = 0,
        future_discount: Optional[Union[float, Dict[str, float]]] = 0.0,
        future_holiday: Optional[Union[int, Dict[str, int]]] = 0,
        future_opening_stock: Optional[Union[float, Dict[str, float]]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Forecast for a specific product. If product_id is None, we forecast for the first product.
        """
        if self.model is None or self.last_row is None:
            raise ValueError("Model not trained. Call train() first.")
        if days < 1:
            raise ValueError("days must be ≥ 1")

        # Determine which product to forecast
        if product_id is None:
            product_id = self.last_row["product_id"].iloc[0]
        # Filter last_row for that product
        last_row_product = self.last_row[self.last_row["product_id"] == product_id]
        if len(last_row_product) == 0:
            raise ValueError(f"Product {product_id} not found in training data.")
        last_row = last_row_product.iloc[0]

        # Prepare future dictionaries
        if not isinstance(future_promotion, dict):
            future_promotion = {i: future_promotion for i in range(days)}
        if not isinstance(future_discount, dict):
            future_discount = {i: future_discount for i in range(days)}
        if not isinstance(future_holiday, dict):
            future_holiday = {i: future_holiday for i in range(days)}
        if future_opening_stock is None:
            # default: use last opening stock
            future_opening_stock = {i: last_row["Opening_Stock_Units"] for i in range(days)}
        elif not isinstance(future_opening_stock, dict):
            future_opening_stock = {i: future_opening_stock for i in range(days)}

        # Get historical demand for this product
        hist_data = self.df_features[self.df_features["product_id"] == product_id]
        last_date = hist_data["Date"].max()
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=days, freq="D"
        )

        # Use demand history for lags
        hist_demand = hist_data["Demand_Units"].tolist()
        # Also need last stock and other features for continuation
        last_stock = last_row["Opening_Stock_Units"]
        last_promo = last_row["promotion"]
        last_discount = last_row["discount_percent"]
        last_holiday = last_row["is_holiday"]

        results = []

        for i, future_date in enumerate(forecast_dates):
            # Build virtual history (actual + predictions)
            virtual_history = hist_demand + [r["predicted_demand"] for r in results]
            idx = len(hist_demand) + i

            # Basic time features
            row = {
                "day_of_week": future_date.dayofweek,
                "day_of_month": future_date.day,
                "month": future_date.month,
                "week_of_year": int(future_date.isocalendar().week),
                "is_weekend": 1 if future_date.dayofweek >= 5 else 0,
                "is_month_start": 1 if future_date.day <= 5 else 0,
                "is_month_end": 1 if future_date.day >= 25 else 0,
            }

            # Lags
            row["demand_lag_1"] = virtual_history[idx - 1] if idx > 0 else hist_demand[-1]
            row["demand_lag_2"] = virtual_history[max(0, idx - 2)]
            row["demand_lag_3"] = virtual_history[max(0, idx - 3)]
            row["demand_lag_7"] = virtual_history[max(0, idx - 7)]

            # Rolling stats
            w3 = virtual_history[max(0, idx - 3): idx]
            w7 = virtual_history[max(0, idx - 7): idx]
            row["demand_roll_mean_3"] = float(np.mean(w3)) if w3 else row["demand_lag_1"]
            row["demand_roll_mean_7"] = float(np.mean(w7)) if w7 else row["demand_lag_1"]
            row["demand_roll_std_3"] = float(np.std(w3)) if len(w3) > 1 else 0.0
            row["demand_roll_std_7"] = float(np.std(w7)) if len(w7) > 1 else 0.0

            # Future inputs
            promo = future_promotion.get(i, 0)
            discount = future_discount.get(i, 0.0)
            holiday = future_holiday.get(i, 0)
            opening_stock = future_opening_stock.get(i, last_stock)

            row["promotion"] = promo
            row["discount_percent"] = discount
            row["is_holiday"] = holiday
            row["Opening_Stock_Units"] = opening_stock
            row["stock_to_demand_ratio"] = opening_stock / (row["demand_roll_mean_7"] + 1)

            # product_id as categorical
            row["product_id"] = product_id

            # Build DataFrame for prediction
            X_future = pd.DataFrame([row])[self.FEATURE_COLS]
            # Ensure categorical columns are string type
            for c in self.CATEGORICAL_FEATURES:
                if c in X_future.columns:
                    X_future[c] = X_future[c].astype(str)

            pred = float(self.model.predict(X_future)[0])

            results.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "day_name": future_date.strftime("%A"),
                "product_id": product_id,
                "promotion": promo,
                "discount_percent": discount,
                "is_holiday": holiday,
                "opening_stock": opening_stock,
                "predicted_demand": round(pred, 2),
            })

        return {product_id: results}

    # --------------------------------------------------------------------- #
    #  Model persistence
    # --------------------------------------------------------------------- #
    def save_model(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained.")
        self.model.save_model(path)

    def load_model(self, path: str) -> None:
        self.model = CatBoostRegressor()
        self.model.load_model(path)
        self._log(f"Model loaded from {path}")