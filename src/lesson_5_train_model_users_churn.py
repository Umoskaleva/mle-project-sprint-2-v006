# ============================================================
# ОБУЧЕНИЕ МОДЕЛИ: ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ
# ============================================================
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# 1. ИМПОРТИРУЕМ ФУНКЦИЮ ИЗ ФАЙЛА
from lesson_5_preprocessing_users_churn import load_clean_users_churn

# ============================================================
# КОНСТАНТЫ
# ============================================================
TABLE_NAME = 'clean_users_churn'
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000
EXPERIMENT_NAME = 'churn_uliana'
RUN_NAME = "logreg_no_encoders_no_leak"

TARGET_COL = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42
DROP_COLS = ["id", "customer_id", "begin_date", "end_date", TARGET_COL]

# ============================================================
# 2. ИНИЦИАЛИЗАЦИЯ MLFLOW
# ============================================================
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
experiment = mlflow.set_experiment(EXPERIMENT_NAME)
experiment_id = experiment.experiment_id
print(f"✅ MLflow эксперимент '{EXPERIMENT_NAME}' инициализирован (ID: {experiment_id})")

# ============================================================
# 3. ПОЛУЧЕНИЕ ДАННЫХ (raw_df)
# ============================================================
# Вызываем функцию, она возвращает DataFrame. 
# .copy() создает независимую копию для безопасных преобразований.
raw_df = load_clean_users_churn(TABLE_NAME).copy()
print(f"✅ Данные загружены. Размер raw_df: {raw_df.shape}")

# Разделяем на признаки (X) и целевую переменную (y)
X = raw_df.copy()
y = raw_df[TARGET_COL].copy()

# Маппинг таргета (если он вдруг строковый)
if y.dtype == object or str(y.dtype).startswith("category"):
    y = y.astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    )
    if y.isna().any():
        raise ValueError(f"Не удалось замапить таргет: {raw_df.loc[y.isna(), TARGET_COL].unique()}")
y = y.astype(int)

# Построчные преобразования (БЕЗ статистик, можно делать ДО сплита)
if "total_charges" in X.columns:
    X["total_charges"] = pd.to_numeric(X["total_charges"], errors="coerce")

if {"begin_date", "end_date"}.issubset(X.columns):
    X["tenure_days"] = (
        pd.to_datetime(X["end_date"], errors="coerce")
        - pd.to_datetime(X["begin_date"], errors="coerce")
    ).dt.days

# Удаляем служебные колонки ПОСЛЕ того, как создали tenure_days
X = X.drop(columns=[col for col in DROP_COLS if col in X.columns])

# ============================================================
# 4. TRAIN/TEST SPLIT (СТРОГО ДО любых fit-операций!)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# ============================================================
# 5. ПОДГОТОВКА ПРИЗНАКОВ БЕЗ УТЕЧЕК
# ============================================================
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

# 5.1. Числовые: медианы считаем ТОЛЬКО по train
train_medians = X_train[num_cols].median()
X_train[num_cols] = X_train[num_cols].fillna(train_medians)
X_test[num_cols]  = X_test[num_cols].fillna(train_medians)

# 5.2. Категориальные: pandas get_dummies + выравнивание колонок
for c in cat_cols:
    X_train[c] = X_train[c].astype(str)
    X_test[c]  = X_test[c].astype(str)

if cat_cols:
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True, dtype=int)
    X_test  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=True, dtype=int)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

# 5.3. Масштабирование числовых признаков
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols]) # fit + transform
X_test[num_cols]  = scaler.transform(X_test[num_cols])      # только transform

X_train = X_train.astype(float)
X_test  = X_test.astype(float)

# ============================================================
# 6. ОБУЧЕНИЕ МОДЕЛИ
# ============================================================
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)

# ============================================================
# 7. ОЦЕНКА
# ============================================================
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "test_accuracy":  float(accuracy_score(y_test, y_pred)),
    "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "test_recall":    float(recall_score(y_test, y_pred, zero_division=0)),
    "test_f1":        float(f1_score(y_test, y_pred, zero_division=0)),
    "test_roc_auc":   float(roc_auc_score(y_test, y_proba)),
}

print("\n📊 Метрики модели:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# ============================================================
# 8. ЛОГИРОВАНИЕ В MLFLOW
# ============================================================
with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as lr_run:
    
    mlflow.log_params({
        "model": "LogisticRegression",
        "max_iter": 3000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "target": TARGET_COL,
        "n_features": X_train.shape[1],
    })
    mlflow.log_metrics(metrics)

    # Сохраняем артефакты для будущего инференса (деплоя)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        joblib.dump(scaler, tmp_path / "scaler.pkl")
        mlflow.log_artifact(str(tmp_path / "scaler.pkl"), artifact_path="preprocessing")
        
        joblib.dump(num_cols, tmp_path / "num_cols.pkl")
        mlflow.log_artifact(str(tmp_path / "num_cols.pkl"), artifact_path="preprocessing")
        
        joblib.dump(train_medians.to_dict(), tmp_path / "train_medians.pkl")
        mlflow.log_artifact(str(tmp_path / "train_medians.pkl"), artifact_path="preprocessing")

    mlflow.sklearn.log_model(model, "logreg_model")

    with tempfile.TemporaryDirectory() as tmpdir:
        pred_path = Path(tmpdir) / "test_predictions.parquet"
        pd.DataFrame({
            "y_true":  y_test.values,
            "y_pred":  y_pred,
            "y_proba": y_proba,
        }).to_parquet(pred_path, index=False)
        mlflow.log_artifact(str(pred_path), artifact_path="predictions")

print(f"\n✅ Запуск успешно залогирован в MLflow! Run ID: {lr_run.info.run_id}")