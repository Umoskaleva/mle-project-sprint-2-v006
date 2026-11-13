# ============================================================
# lesson_8_train_model_users_churn_autofeat.py
# Задача оттока клиентов
# Обучение LogisticRegression с автогенерацией признаков (AutoFeat)
# Кодирование категориальных признаков происходит ВНУТРИ AutoFeat
# Регистрация модели в MLflow Model Registry.
# ============================================================


import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


import sklearn.utils
import sklearn.utils.validation
from sklearn.utils import check_array as _original_check_array


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


from lesson_5_preprocessing_users_churn import load_clean_users_churn



# ============================================================
# ПАТЧИ СОВМЕСТИМОСТИ
# ============================================================
# autofeat использует:
# 1. Series.ravel(), которого нет в новых pandas.
# 2. force_all_finite, который в новых версиях scikit-learn
#    переименован в ensure_all_finite.
#
# ВАЖНО: патчи должны быть выполнены ДО импорта AutoFeatClassifier.
# ============================================================


if not hasattr(pd.Series, "ravel"):
    def _series_ravel(self):
        return self.to_numpy().ravel()


    pd.Series.ravel = _series_ravel



def _check_array_compat(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")


    return _original_check_array(*args, **kwargs)



# Патчим sklearn до импорта autofeat
sklearn.utils.check_array = _check_array_compat
sklearn.utils.validation.check_array = _check_array_compat



# Импортируем autofeat только после патчей
from autofeat import AutoFeatClassifier


# Дополнительно подменяем ссылку непосредственно в модуле autofeat.
# Это гарантирует, что transform() будет использовать обёртку.
import autofeat.autofeat as _autofeat_module


_autofeat_module.check_array = _check_array_compat



# ============================================================
# КОНСТАНТЫ
# ============================================================
TABLE_NAME = "clean_users_churn"
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000
EXPERIMENT_NAME = "churn_uliana"
RUN_NAME = "logreg_with_autofeat"
REGISTRY_MODEL_NAME = "logreg_model_with_autofeat"
AFC_REGISTRY_MODEL_NAME = "afc_model"


TARGET_COL = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42


cat_features = [
    "type",
    "payment_method",
    "internet_service",
    "gender",
    "paperless_billing",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "partner",
    "dependents",
    "multiple_lines",
]


num_features = [
    "monthly_charges",
    "total_charges",
    "senior_citizen",
]


features = cat_features + num_features
split_column = "begin_date"



# ============================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================
df = load_clean_users_churn(TABLE_NAME).copy()


for col in num_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")


df[num_features] = df[num_features].fillna(df[num_features].median())


for col in cat_features:
    df[col] = df[col].fillna("unknown").astype(str)


df = df.sort_values(by=[split_column])



# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    df[features],
    df[TARGET_COL],
    test_size=TEST_SIZE,
    shuffle=False,
)



# ============================================================
# 3. AUTOFEAT + LOGISTIC REGRESSION В ОДНОМ PIPELINE
# ============================================================
transformations = ("1/", "log", "abs", "sqrt")


# ВАЖНО: AutoFeatClassifier создаётся ДО Pipeline.
# У AutoFeatClassifier нет параметра random_state.
autofeat = AutoFeatClassifier(
    categorical_cols=cat_features,
    transformations=transformations,
    feateng_steps=1,
    n_jobs=1,
)


model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=RANDOM_STATE,
)


pipeline = Pipeline(
    steps=[
        ("autofeat", autofeat),
        ("logistic_regression", model),
    ]
)


pipeline.fit(X_train, y_train)



# ============================================================
# 4. МЕТРИКИ
# ============================================================
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]


metrics = {
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
}



# ============================================================
# 5. MLFLOW: ЭКСПЕРИМЕНТ, ПАРАМЕТРЫ, МЕТРИКИ, МОДЕЛИ
# ============================================================
tracking_uri = f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"


mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)


experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)


if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id



with mlflow.start_run(
    run_name=RUN_NAME,
    experiment_id=experiment_id,
) as run:


    run_id = run.info.run_id


    mlflow.log_params(
        {
            "model": "LogisticRegression",
            "feature_engineering": "AutoFeatClassifier",
            "autofeat_transformations": str(transformations),
            "autofeat_feateng_steps": 1,
            "autofeat_n_jobs": 1,
            "n_features_before_autofeat": X_train.shape[1],
            "n_features_after_autofeat": pipeline.named_steps["autofeat"].transform(X_train).shape[1],
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        }
    )


    mlflow.log_metrics(metrics)


    # ------------------------------------------------------------
    # ОТДЕЛЬНО ЛОГИРУЕМ И РЕГИСТРИРУЕМ AUTOFEAT
    # ------------------------------------------------------------
    afc_info = mlflow.sklearn.log_model(
        sk_model=autofeat,
        artifact_path="afc",
        registered_model_name=AFC_REGISTRY_MODEL_NAME,
        serialization_format="cloudpickle",
    )


    # ------------------------------------------------------------
    # ЛОГИРУЕМ И РЕГИСТРИРУЕМ ПОЛНЫЙ PIPELINE
    # ------------------------------------------------------------
    signature = infer_signature(
        X_train,
        pipeline.predict(X_train),
    )


    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="churn_model",
        signature=signature,
        input_example=X_train.head(5),
        registered_model_name=REGISTRY_MODEL_NAME,
        serialization_format="cloudpickle",
    )


    print(f"Run ID: {run_id}")
    print(f"AFC Model URI: {afc_info.model_uri}")
    print(f"AFC Registered model: {AFC_REGISTRY_MODEL_NAME}")
    print(f"Model URI: {model_info.model_uri}")
    print(f"Registered model: {REGISTRY_MODEL_NAME}")