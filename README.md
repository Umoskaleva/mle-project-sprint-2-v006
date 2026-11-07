# Проект: churn_model_uliana (mle-project-sprint-2-v006)

## Описание
Проект по работе с моделью предсказания оттока клиентов (**churn classification**).
Модель обучается, логируется и регистрируется в **MLflow**, который поднимается локально.
Артефакты модели хранятся локально. 

## Структура проекта
```
# mle-project-sprint-2-v006

## 📂 Структура проекта

```text
.
├── mlflow_local/
│   └── artifacts_users_charn/
│       └── 1/                      # Артефакты экспериментов MLflow
├── notebooks/
│   └── postgres_connection_mybase.ipynb  # Ноутбук с подключением к БД и анализом
├── scripts/
│   ├── run_mlflow_server_local_users_churn.sh   # Скрипт запуска локального сервера MLflow
│   └── run_mlflow_server_postgresql.sh          # Скрипт запуска MLflow с бэкендом PostgreSQL
├── src/                            # Исходный код проекта
│   └── ...
├── .gitignore                      # Файлы и папки, исключаемые из Git
├── README.md                       # Документация проекта
├── params.yaml                     # Гиперпараметры и конфигурация модели
├── python_env.yaml                 # Описание окружения Python (версия, зависимости)
└── requirements.txt                # Список Python-библиотек для установки
```

## Установка
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Создайте .env с переменными:
MLFLOW_TRACKING_URI=<URL MLflow-сервера>
MLFLOW_S3_ENDPOINT_URL=<URL S3>
AWS_ACCESS_KEY_ID=<ключ>
AWS_SECRET_ACCESS_KEY=<секрет>


## Запуск MLflow локально
```
run_mlflow_server_local_users_churn.sh
```


## 1. Постановка задачи

Требовалось:

1. Обучить новую версию модели `churn_model_uliana` (v2) используя энкодеры для признаков. 
2. Зарегистрировать её в реестре моделей MLflow.
3. Посчитать метрики: ROC-AUC, Precision, Recall, F1, Accuracy.
4. Сравнить v1 и v2 между собой.
