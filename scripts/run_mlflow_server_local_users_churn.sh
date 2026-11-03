mkdir -p mlflow_local/artifacts_users_charn
mlflow server \
  --backend-store-uri sqlite:///./mlflow_local/mlflow.db \
  --default-artifact-root file:./mlflow_local/artifacts_users_charn \
  --host 127.0.0.1 \
  --port 5000