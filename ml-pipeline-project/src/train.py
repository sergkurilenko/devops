"""
Скрипт для обучения модели с отслеживанием в MLflow
"""
import argparse
import os
import warnings
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def load_data():
    """Загрузка датасета California Housing"""
    print("Загрузка данных California Housing...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    # Сохранение данных локально
    os.makedirs("../data", exist_ok=True)
    df.to_csv("../data/california_housing.csv", index=False)
    print(f"Данные сохранены: shape={df.shape}")
    
    return df


def prepare_data(df, test_size=0.2, random_state=42):
    """Подготовка данных для обучения"""
    print("Подготовка данных...")
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_type='rf', **kwargs):
    """Обучение модели"""
    print(f"Обучение модели: {model_type}")
    
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1
        )
    elif model_type == 'lr':
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    print("Модель обучена!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Оценка модели"""
    print("Оценка модели...")
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    print(f"Метрики:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return metrics, y_pred


def save_predictions(y_test, y_pred, filename='predictions.csv'):
    """Сохранение предсказаний"""
    os.makedirs("../reports", exist_ok=True)
    
    results = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred,
        'error': y_test.values - y_pred
    })
    
    filepath = f"../reports/{filename}"
    results.to_csv(filepath, index=False)
    print(f"Предсказания сохранены: {filepath}")
    
    return filepath


def main(args):
    """Основная функция"""
    print("="*50)
    print("ЗАПУСК ML PIPELINE С MLFLOW")
    print("="*50)
    
    # Настройка MLflow
    experiment_name = "California_Housing_Prediction"
    mlflow.set_experiment(experiment_name)
    
    # Загрузка и подготовка данных
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Начало MLflow run
    with mlflow.start_run(run_name=f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Логирование параметров
        params = {
            'model_type': args.model_type,
            'test_size': 0.2,
            'random_state': 42
        }
        
        if args.model_type == 'rf':
            params['n_estimators'] = args.n_estimators
            params['max_depth'] = args.max_depth
        
        mlflow.log_params(params)
        
        # Обучение модели
        model = train_model(
            X_train, y_train,
            model_type=args.model_type,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        
        # Оценка модели
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        
        # Логирование метрик
        mlflow.log_metrics(metrics)
        
        # Сохранение предсказаний
        pred_file = save_predictions(y_test, y_pred)
        mlflow.log_artifact(pred_file)
        
        # Сохранение модели
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=f"{args.model_type}_california_housing"
        )
        
        # Сохранение модели локально
        os.makedirs("../models", exist_ok=True)
        model_path = f"../models/{args.model_type}_model.pkl"
        import joblib
        joblib.dump(model, model_path)
        print(f"Модель сохранена локально: {model_path}")
        
        # Дополнительная информация
        mlflow.set_tag("developer", "Sergey")
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.set_tag("dataset", "california_housing")
        
        print("\n" + "="*50)
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"Experiment ID: {mlflow.active_run().info.experiment_id}")
        print("="*50)
    
    print("\nГотово! Запустите 'mlflow ui' для просмотра результатов")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model with MLflow")
    parser.add_argument(
        '--model-type',
        type=str,
        default='rf',
        choices=['rf', 'lr'],
        help='Model type: rf (Random Forest) or lr (Linear Regression)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of trees for Random Forest'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=10,
        help='Maximum depth for Random Forest'
    )
    
    args = parser.parse_args()
    main(args)
