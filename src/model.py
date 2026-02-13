import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from data_cleaning import load_data, clean_data
from feature_engineering import build_preprocessor


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """
    Train model, perform cross validation,
    and evaluate on test set
    """

    cv_rmse = -cross_val_score(
        model,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=5
    ).mean()

    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return cv_rmse, rmse, mae, r2


def train_models(data_path):

    df = load_data(data_path)
    df = clean_data(df)

    X = df.drop("Price", axis=1)

    y = np.log1p(df["Price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(df)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=5000),
        "Random Forest": RandomForestRegressor(random_state=42)
    }

    results = []

    best_model = None
    best_rmse = float("inf")

    print("\n🔹 Training Base Models...\n")

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        cv_rmse, rmse, mae, r2 = evaluate_model(
            pipeline, pipeline,
            X_train, y_train,
            X_test, y_test
        )

        results.append([name, cv_rmse, rmse, mae, r2])

        print(f"{name}")
        print(f"CV RMSE: {cv_rmse:.2f}")
        print(f"Test RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R2: {r2:.4f}")
        print("-" * 40)

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline

    print("\n🔹 Hyperparameter Tuning (Random Forest)...\n")

    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        rf_pipeline,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_

    cv_rmse, rmse, mae, r2 = evaluate_model(
        best_rf, best_rf,
        X_train, y_train,
        X_test, y_test
    )

    results.append(["Tuned Random Forest", cv_rmse, rmse, mae, r2])

    print("Best RF Parameters:", grid.best_params_)
    print(f"Tuned RF RMSE: {rmse:.2f}")
    print("-" * 40)

    if rmse < best_rmse:
        best_model = best_rf
    
    results_df = pd.DataFrame(
        results,
        columns=["Model", "CV_RMSE", "Test_RMSE", "MAE", "R2"]
    ).sort_values("Test_RMSE")

    print("\n📊 Model Performance Comparison:\n")
    print(results_df)

    joblib.dump(best_model, "best_model.pkl")
    print("\n✅ Best model saved as best_model.pkl")

    return results_df


if __name__ == "__main__":
    train_models("data/raw/MagicBricks.csv")
