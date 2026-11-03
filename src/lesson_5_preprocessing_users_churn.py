# Подключиться к PostgreSQL через параметры из .env.
# Загрузить таблицу clean_users_churn в pandas.
# Закодировать категориальные признаки через One-Hot.
# Из monthly_charges и total_charges создать несколько версий числовых признаков.
# Собрать преобразования в ColumnTransformer.
# Применить его к DataFrame и сохранить препроцессор в MLflow.



import os

import pandas as pd
import mlflow
import tempfile
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
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

TABLE_NAME = 'clean_users_churn'

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

EXPERIMENT_NAME = 'churn_uliana'
RUN_NAME = "preprocessing"
REGISTRY_MODEL_NAME = 'uliana_users_churn'


def get_destination_engine():
    """Engine для DB_DESTINATION_*. Читаем users_churn отсюда."""
    load_dotenv()
    host = os.getenv("DB_DESTINATION_HOST")
    port = os.getenv("DB_DESTINATION_PORT")
    db_name = os.getenv("DB_DESTINATION_NAME")
    username = os.getenv("DB_DESTINATION_USER")
    password = os.getenv("DB_DESTINATION_PASSWORD")
    if not all([host, port, db_name, username, password]):
        raise ValueError("Не все переменные окружения для destination заданы")

    dsn = (
        f"postgresql+psycopg2://{username}:{password}"
        f"@{host}:{port}/{db_name}?sslmode=disable"
    )
    engine = create_engine(dsn)

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"✅ Подключение к destination '{db_name}' на {host}:{port}")
    except Exception as e:
        print(f"❌ Не удалось подключиться к destination: {e}")
        engine.dispose()
        raise

    return engine


def load_clean_users_churn(table_name: str = TABLE_NAME) -> pd.DataFrame:
    """Загружает таблицу clean_users_churn из DB_DESTINATION."""
    engine = get_destination_engine()
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        print(f"✅ Загружено {len(df)} строк из '{table_name}'")
        return df
    except Exception as e:
        print(f"❌ Ошибка загрузки '{table_name}': {e}")
        raise
    finally:
        engine.dispose()


# Загрузка данных в переменную df
df = load_clean_users_churn()
raw_df = df.copy()   # <-- сохраняем "сырую" копию до всех преобразований

print(df.head())
print(df.info())


# определение категориальных колонок, которые будут преобразованы
cat_columns = ["type", "payment_method", "internet_service", "gender"]

# создание объекта OneHotEncoder для преобразования категориальных переменных
# auto - автоматическое определение категорий
# ignore - игнорировать ошибки, если встречается неизвестная категория
# max_categories - максимальное количество уникальных категорий
# sparse_output - вывод в виде разреженной матрицы, если False, то в виде обычного массива
# drop="first" - удаляет первую категорию, чтобы избежать ловушки мультиколлинеарности
encoder_oh = OneHotEncoder(
    categories='auto',
    handle_unknown='ignore',
    max_categories=10,
    sparse_output=False,
    drop='first'
)

# применение OneHotEncoder к данным. Преобразование категориальных данных в массив
encoded_features = encoder_oh.fit_transform(df[cat_columns].to_numpy())

# преобразование полученных признаков в DataFrame и установка названий колонок
# get_feature_names_out() - получение имён признаков после преобразования
encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder_oh.get_feature_names_out(cat_columns),
    index=df.index
)

# конкатенация исходного DataFrame с новым DataFrame, содержащим закодированные категориальные признаки
# axis=1 означает конкатенацию по колонкам
df = pd.concat([df, encoded_df], axis=1)

df.head(2)


num_columns = ["monthly_charges", "total_charges"]

n_knots = 3
degree_spline = 4
n_quantiles = 100
degree = 3
n_bins = 5
encode = 'ordinal'
strategy = 'uniform'
subsample = None

# SplineTransformer
encoder_spl = SplineTransformer(
    n_knots=n_knots,
    degree=degree_spline
)
encoded_features = encoder_spl.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder_spl.get_feature_names_out(num_columns),
    index=df.index
)
df = pd.concat([df, encoded_df], axis=1)

# QuantileTransformer
encoder_q = QuantileTransformer(n_quantiles=n_quantiles)
encoded_features = encoder_q.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder_q.get_feature_names_out(num_columns),
    index=df.index
)
encoded_df.columns = [col + f"_q_{n_quantiles}" for col in num_columns]
df = pd.concat([df, encoded_df], axis=1)

# RobustScaler
encoder_rb = RobustScaler()
encoded_features = encoder_rb.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder_rb.get_feature_names_out(num_columns),
    index=df.index
)
encoded_df.columns = [col + f"_robust" for col in num_columns]
df = pd.concat([df, encoded_df], axis=1)

# PolynomialFeatures
encoder_pol = PolynomialFeatures(degree=degree)
encoded_features = encoder_pol.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder_pol.get_feature_names_out(num_columns),
    index=df.index
)
# Убираем первый столбец (константу 1) и исходные признаки,
# оставляем только новые полиномиальные признаки
encoded_df = encoded_df.drop(columns=['1'])  # убрать только константу
encoded_df.columns = [f"{col}_poly" for col in encoded_df.columns]
df = pd.concat([df, encoded_df], axis=1)

# KBinsDiscretizer
encoder_kbd = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy, subsample=subsample)
encoded_features = encoder_kbd.fit_transform(df[num_columns].to_numpy())

encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder_kbd.get_feature_names_out(num_columns),
    index=df.index
)
encoded_df.columns = [col + f"_bin" for col in num_columns]
df = pd.concat([df, encoded_df], axis=1)

df.head(2)


numeric_transformer = ColumnTransformer(transformers=[
    ('spl', encoder_spl, num_columns),
    ('q', encoder_q, num_columns),
    ('rb', encoder_rb, num_columns),
    ('pol', encoder_pol, num_columns),
    ('kbd', encoder_kbd, num_columns),
])

# категориальные преобразования
categorical_transformer = Pipeline(steps=[('encoder', encoder_oh)])

# объединяем
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_columns),
    ('cat', categorical_transformer, cat_columns),
], n_jobs=-1)

encoded_features = preprocessor.fit_transform(df)

transformed_df = pd.DataFrame(
    encoded_features,
    columns=preprocessor.get_feature_names_out(),
    index=df.index
)

df = pd.concat([df, transformed_df], axis=1)
df.head(2)



os.environ["MLFLOW_S3_ENDPOINT_URL"] = "MLFLOW_S3_ENDPOINT_URL"
os.environ["AWS_ACCESS_KEY_ID"] = "AWS_ACCESS_KEY_ID"
os.environ["AWS_SECRET_ACCESS_KEY"] = "AWS_SECRET_ACCESS_KEY"

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    # Эксперимента нет — создаём его
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

# 1. Логируем сам препроцессор
    mlflow.sklearn.log_model(
    preprocessor,
    "column_transformer",
    skops_trusted_types=[
        "scipy.interpolate._bsplines.BSpline",
        "scipy.interpolate._bsplines._BSpline"
    ]
    )
    
# 2. Логируем "сырой" и обработанный df
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        raw_path = tmpdir / "raw_df.parquet"
        proc_path = tmpdir / "processed_df.parquet"

        raw_df.to_parquet(raw_path, index=False)
        df.to_parquet(proc_path, index=False) 

        mlflow.log_artifact(str(raw_path), artifact_path="data")
        mlflow.log_artifact(str(proc_path), artifact_path="data")
    