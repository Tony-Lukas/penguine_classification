import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pickle

# hyper-parameters
C_param = 1.0
kernel_param = "rbf"

# Set the MLflow expriment name
mlflow.set_experiment("Penguins_SVM_Classification")

with mlflow.start_run():
    data_path = "data/penguins_size.csv"
    df = pd.read_csv(data_path)
    # data preprocessing
    df = df.dropna()
    y = df["species"]
    X = df.drop(columns=["species"])
    # X = pd.get_dummies(X, columns=["island", "sex"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    pickle.dump(le, open("le.pkl", "wb"))

    # pipline
    num_col = X_train.select_dtypes(include="number").columns
    cat_col = X_train.select_dtypes(include="object").columns

    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]
    )
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_col),
            ("cat", cat_transformer, cat_col),
        ],
        remainder="drop",
    )

    # log hyper-parameters
    mlflow.log_param("C", C_param)
    mlflow.log_param("kernel", kernel_param)
    mlflow.log_param("data_path", data_path)
    mlflow.log_param("train_size", len(X_train))

    # model train
    model = SVC(C=C_param, kernel=kernel_param, random_state=42)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    # log matrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", pre)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    print(f"Model Accuracy: {acc:.4f}")
    # log model Artifact
    mlflow.sklearn.log_model(
        pipeline, name="svm_full_pipeline", input_example=X_test[0:5]
    )

    mlflow.set_tag("git_commit", os.popen("git rev-parse HEAD").read().strip())

    import dvc.api

    try:
        dvc_version = dvc.api.data_cloud.get_url(data_path, checksum=True).split("?")[1]
        mlflow.log_param("dvc_data_version", dvc_version)
    except Exception as e:
        mlflow.log_param("dvc_data_version", "Could not retrieve DVC version")
