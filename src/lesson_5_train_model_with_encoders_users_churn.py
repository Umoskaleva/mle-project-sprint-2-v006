# ============================================================
# lesson_6_logreg_with_encoders.py
# Обучение LogisticRegression на признаках, полученных энкодерами.
# Без утечки: препроцессор обучается только на X_train.
# Регистрация модели в MLflow Model Registry (локальные артефакты).
# ============================================================

import os
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    SplineTransformer,
    QuantileTransformer,
    RobustScaler,
    PolynomialFeatures,
    KBinsDiscretizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ============================================================
# 1. ИМПОРТИРУЕМ ФУНКЦИЮ ИЗ ФАЙЛА
# ============================================================
from lesson_5_preprocessing_users_churn import load_clean_users_churn


# ============================================================
# КОНСТАНТЫ
# ============================================================
TABLE_NAME = 'clean_users_churn'
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000
EXPERIMENT_NAME = 'churn_uliana'
RUN_NAME = "logreg_with_encoders_no_leak"
REGISTRY_MODEL_NAME = 'logreg_model_with_encoders'

TARGET_COL = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# оставляем как есть: id/customer_id/даты не используем, target — цель
DROP_COLS = ["id", "customer_id", "begin_date", "end_date", TARGET_COL]

# категориальные признаки
CAT_COLUMNS = [
    "type", "payment_method", "internet_service", "gender",
    "paperless_billing",
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
    "partner", "dependents", "multiple_lines",
]

# числовые признаки (senior_citizen — 0/1, кладём сюда же)
NUM_COLUMNS = ["monthly_charges", "total_charges", "senior_citizen"]


# ============================================================
# ПРЕПРОЦЕССОР
# ============================================================
def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer с энкодерами для числовых и категориальных признаков."""

    # --- параметры энкодеров ---
    n_knots = 3
    degree_spline = 4
    n_quantiles = 100
    degree = 3
    n_bins = 5
    encode = 'ordinal'
    strategy = 'uniform'
    subsample = None

    # --- категориальные ---
    encoder_oh = OneHotEncoder(
        categories='auto',
        handle_unknown='ignore',
        max_categories=10,
        sparse_output=False,
        drop='first',
    )

    # --- числовые ---
    encoder_spl = SplineTransformer(
        n_knots=n_knots,
        degree=degree_spline,
    )
    encoder_q = QuantileTransformer(n_quantiles=n_quantiles)
    encoder_rb = RobustScaler()
    encoder_pol = PolynomialFeatures(degree=degree, include_bias=False)
    encoder_kbd = KBinsDiscretizer(
        n_bins=n_bins, encode=encode, strategy=strategy, subsample=subsample,
    )

    numeric_transformer = ColumnTransformer(
        transformers=[
            ('spl', encoder_spl, NUM_COLUMNS),
            ('q',   encoder_q,   NUM_COLUMNS),
            ('rb',  encoder_rb,  NUM_COLUMNS),
            ('pol', encoder_pol, NUM_COLUMNS),
            ('kbd', encoder_kbd, NUM_COLUMNS),
        ],
        remainder='drop',
    )

    categorical_transformer = Pipeline(steps=[('encoder', encoder_oh)])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUM_COLUMNS),
            ('cat', categorical_transformer, CAT_COLUMNS),
        ],
        remainder='drop',
        n_jobs=-1,
    )


# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================
raw_df = load_clean_users_churn(TABLE_NAME).copy()
print(f"✅ Данные загружены. Размер raw_df: {raw_df.shape}")

y = raw_df[TARGET_COL].astype(int)
X = raw_df.drop(columns=DROP_COLS, errors="ignore")

# приведение числовых (на случай строк в БД) + медианная импутация
for col in NUM_COLUMNS:
    X[col] = pd.to_numeric(X[col], errors="coerce")
X[NUM_COLUMNS] = X[NUM_COLUMNS].fillna(X[NUM_COLUMNS].median())

# ---------- 2. Проверка, что ничего не потерялось ----------
expected_used = set(CAT_COLUMNS) | set(NUM_COLUMNS)
missing = expected_used - set(X.columns)
assert not missing, f"В X отсутствуют колонки: {missing}"

unused = set(X.columns) - expected_used
print(f"ℹ️  Используется признаков: {len(expected_used)}")
print(f"ℹ️  Не используется ({len(unused)}): {sorted(unused)}")
assert not unused, f"Признаки не попали ни в CAT, ни в NUM: {unused}"

# ---------- 3. Train/test split ДО fit препроцессора ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)
print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

# ---------- 4. Pipeline + обучение ----------
pipeline = Pipeline(steps=[
    ('preprocessor', build_preprocessor()),
    ('model', LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )),
])

# fit только на train — модель видит признаки, полученные энкодерами
pipeline.fit(X_train, y_train)

# ---------- 5. Метрики ----------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "roc_auc":   float(roc_auc_score(y_test, y_proba)),
    "accuracy":  float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
    "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
}
print("✅ Метрики на тесте:", metrics)
print(classification_report(y_test, y_pred))

# ---------- 6. MLflow ----------
# локальные артефакты:
# mlflow server --backend-store-uri sqlite:///./mlflow_local/mlflow.db \
#               --default-artifact-root file:./mlflow_local/artifacts_users_charn

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    # параметры
    mlflow.log_params({
        "model": "LogisticRegression",
        "solver": "lbfgs",
        "max_iter": 3000,
        "class_weight": "balanced",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_features_in": X_train.shape[1],
        "n_features_out": pipeline.named_steps["preprocessor"]
                                 .get_feature_names_out().shape[0],
        "cat_columns": CAT_COLUMNS,
        "num_columns": NUM_COLUMNS,
    })

    # метрики
    mlflow.log_metrics(metrics)

    # препроцессор отдельно (skops — из-за BSpline)
    mlflow.sklearn.log_model(
        pipeline.named_steps["preprocessor"],
        artifact_path="column_transformer",
        skops_trusted_types=[
            "scipy.interpolate._bsplines.BSpline",
            "scipy.interpolate._bsplines._BSpline",
        ],
    )

    # полный Pipeline — регистрируем в Model Registry
    model_info = mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name=REGISTRY_MODEL_NAME,
        skops_trusted_types=[
            "scipy.interpolate._bsplines.BSpline",
            "scipy.interpolate._bsplines._BSpline",
        ],
    )
    print(f"✅ Модель зарегистрирована: {REGISTRY_MODEL_NAME}, run_id={run_id}")
    print(f"✅ model_uri = {model_info.model_uri}")

    # артефакты данных (только transform, без fit — утечки нет)
    transformed_test = pipeline.named_steps["preprocessor"].transform(X_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        raw_path     = tmpdir / "raw_df_test.parquet"
        proc_path    = tmpdir / "processed_X_test.parquet"
        metrics_path = tmpdir / "metrics.json"
        report_path  = tmpdir / "classification_report.txt"

        X_test.assign(**{TARGET_COL: y_test.values}).to_parquet(raw_path, index=False)
        pd.DataFrame(
            transformed_test,
            columns=pipeline.named_steps["preprocessor"].get_feature_names_out(),
        ).to_parquet(proc_path, index=False)

        metrics_path.write_text(json.dumps(metrics, indent=2))
        report_path.write_text(classification_report(y_test, y_pred))

        mlflow.log_artifact(str(raw_path),     artifact_path="data")
        mlflow.log_artifact(str(proc_path),    artifact_path="data")
        mlflow.log_artifact(str(metrics_path), artifact_path="data")
        mlflow.log_artifact(str(report_path),  artifact_path="data")

print("Готово.")