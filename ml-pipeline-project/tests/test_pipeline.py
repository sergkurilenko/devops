"""
Unit-тесты для ML pipeline
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def test_data_loading():
    """Тест загрузки данных"""
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    assert df is not None
    assert len(df) > 0
    assert 'MedHouseVal' in df.columns


def test_data_shape():
    """Тест размерности данных"""
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    assert df.shape[0] > 1000  # Достаточно строк
    assert df.shape[1] == 9    # 8 признаков + 1 целевая переменная


def test_no_missing_values():
    """Тест на отсутствие пропущенных значений"""
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    assert df.isnull().sum().sum() == 0


def test_target_variable():
    """Тест целевой переменной"""
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    y = df['MedHouseVal']
    assert y.min() > 0
    assert y.max() < 10  # Цены в сотнях тысяч


def test_feature_dtypes():
    """Тест типов данных признаков"""
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    X = df.drop('MedHouseVal', axis=1)
    for col in X.columns:
        assert pd.api.types.is_numeric_dtype(X[col])


def test_train_test_split():
    """Тест разделения данных"""
    from sklearn.model_selection import train_test_split
    
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    assert len(X_train) > len(X_test)
    assert len(X_train) + len(X_test) == len(X)


def test_model_training():
    """Тест обучения модели"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train[:100], y_train[:100])  # Быстрое обучение на части данных
    
    assert model is not None
    assert hasattr(model, 'predict')


def test_model_prediction():
    """Тест предсказания модели"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train[:100], y_train[:100])
    
    y_pred = model.predict(X_test[:10])
    
    assert len(y_pred) == 10
    assert all(y_pred > 0)


def test_metrics_calculation():
    """Тест вычисления метрик"""
    from sklearn.metrics import mean_squared_error, r2_score
    
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    assert mse >= 0
    assert -1 <= r2 <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
