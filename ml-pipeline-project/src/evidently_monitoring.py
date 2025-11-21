"""
Мониторинг дрейфа данных с использованием EvidentlyAI
"""
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    RegressionPreset
)
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.tests import *
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def load_data():
    """Загрузка данных"""
    print("Загрузка данных California Housing...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    print(f"Размер данных: {df.shape}")
    return df


def create_reference_and_current_data(df):
    """Создание reference и current датасетов"""
    # Разделение на reference (обучающие) и current (тестовые) данные
    reference_data = df.sample(frac=0.7, random_state=42)
    current_data = df.drop(reference_data.index)
    
    # Добавление prediction колонки для симуляции
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    
    X_ref = reference_data.drop('MedHouseVal', axis=1)
    y_ref = reference_data['MedHouseVal']
    model.fit(X_ref, y_ref)
    
    reference_data['prediction'] = model.predict(X_ref)
    
    X_cur = current_data.drop('MedHouseVal', axis=1)
    current_data['prediction'] = model.predict(X_cur)
    
    print(f"Reference data: {reference_data.shape}")
    print(f"Current data: {current_data.shape}")
    
    return reference_data, current_data


def simulate_data_drift(current_data):
    """Симуляция дрейфа данных для демонстрации"""
    drifted_data = current_data.copy()
    
    # Добавление небольшого дрейфа в признаки
    drifted_data['MedInc'] = drifted_data['MedInc'] * 1.2 + np.random.normal(0, 0.1, len(drifted_data))
    drifted_data['AveRooms'] = drifted_data['AveRooms'] * 1.1
    
    print(f"Drifted data created: {drifted_data.shape}")
    return drifted_data


def create_column_mapping():
    """Создание маппинга колонок"""
    column_mapping = ColumnMapping()
    column_mapping.target = 'MedHouseVal'
    column_mapping.prediction = 'prediction'
    column_mapping.numerical_features = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    return column_mapping


def generate_data_drift_report(reference_data, current_data, column_mapping):
    """Генерация отчета о дрейфе данных"""
    print("\n" + "="*60)
    print("ОТЧЕТ О ДРЕЙФЕ ДАННЫХ (Data Drift Report)")
    print("="*60)
    
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    # Сохранение отчета
    os.makedirs("../reports", exist_ok=True)
    report_path = "../reports/evidently_data_drift.html"
    report.save_html(report_path)
    
    print(f"Отчет сохранен: {report_path}")
    
    # Вывод метрик
    result = report.as_dict()
    if 'metrics' in result and len(result['metrics']) > 0:
        drift_share = result['metrics'][0]['result']['share_of_drifted_columns']
        print(f"\nДоля признаков с дрейфом: {drift_share:.2%}")
    
    return report


def generate_data_quality_report(reference_data, current_data, column_mapping):
    """Генерация отчета о качестве данных"""
    print("\n" + "="*60)
    print("ОТЧЕТ О КАЧЕСТВЕ ДАННЫХ (Data Quality Report)")
    print("="*60)
    
    report = Report(metrics=[
        DataQualityPreset(),
    ])
    
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    report_path = "../reports/evidently_data_quality.html"
    report.save_html(report_path)
    
    print(f"Отчет сохранен: {report_path}")
    return report


def generate_model_performance_report(reference_data, current_data, column_mapping):
    """Генерация отчета о производительности модели"""
    print("\n" + "="*60)
    print("ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛИ (Model Performance)")
    print("="*60)
    
    report = Report(metrics=[
        RegressionPreset(),
    ])
    
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    report_path = "../reports/evidently_model_performance.html"
    report.save_html(report_path)
    
    print(f"Отчет сохранен: {report_path}")
    return report


def run_data_drift_tests(reference_data, current_data, column_mapping):
    """Запуск тестов на дрейф данных"""
    print("\n" + "="*60)
    print("ТЕСТЫ НА ДРЕЙФ ДАННЫХ (Data Drift Tests)")
    print("="*60)
    
    test_suite = TestSuite(tests=[
        DataDriftTestPreset(),
        TestShareOfMissingValues(lte=0.05),
        TestNumberOfConstantColumns(eq=0),
        TestNumberOfEmptyRows(eq=0),
        TestNumberOfEmptyColumns(eq=0),
        TestNumberOfDuplicatedColumns(eq=0),
        TestNumberOfDuplicatedRows(lte=10),
    ])
    
    test_suite.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    # Сохранение отчета
    report_path = "../reports/evidently_drift_tests.html"
    test_suite.save_html(report_path)
    
    print(f"Отчет тестов сохранен: {report_path}")
    
    # Вывод результатов тестов
    result = test_suite.as_dict()
    print("\nРезультаты тестов:")
    if 'tests' in result:
        passed = sum(1 for test in result['tests'] if test['status'] == 'SUCCESS')
        total = len(result['tests'])
        print(f"Пройдено: {passed}/{total}")
        
        # Детали по каждому тесту
        for test in result['tests']:
            status = "✓" if test['status'] == 'SUCCESS' else "✗"
            print(f"  {status} {test['name']}: {test['status']}")
    
    return test_suite


def compare_distributions(reference_data, current_data):
    """Сравнение распределений признаков"""
    print("\n" + "="*60)
    print("СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ ПРИЗНАКОВ")
    print("="*60)
    
    numerical_features = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    
    results = []
    for feature in numerical_features:
        ref_mean = reference_data[feature].mean()
        cur_mean = current_data[feature].mean()
        ref_std = reference_data[feature].std()
        cur_std = current_data[feature].std()
        
        mean_diff = abs(cur_mean - ref_mean) / ref_mean * 100
        std_diff = abs(cur_std - ref_std) / ref_std * 100
        
        results.append({
            'feature': feature,
            'ref_mean': ref_mean,
            'cur_mean': cur_mean,
            'mean_diff_%': mean_diff,
            'std_diff_%': std_diff
        })
        
        print(f"\n{feature}:")
        print(f"  Среднее: {ref_mean:.4f} → {cur_mean:.4f} ({mean_diff:+.2f}%)")
        print(f"  Std:     {ref_std:.4f} → {cur_std:.4f} ({std_diff:+.2f}%)")
    
    # Сохранение в CSV
    results_df = pd.DataFrame(results)
    results_path = "../reports/distribution_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nСравнение сохранено: {results_path}")
    
    return results_df


def main():
    """Основная функция"""
    print("="*60)
    print("EVIDENTLY AI: МОНИТОРИНГ ДРЕЙФА ДАННЫХ")
    print("="*60)
    
    # Загрузка данных
    df = load_data()
    
    # Создание reference и current данных
    reference_data, current_data = create_reference_and_current_data(df)
    
    # Создание маппинга колонок
    column_mapping = create_column_mapping()
    
    # Генерация отчетов
    print("\n" + ">"*60)
    print("ГЕНЕРАЦИЯ ОТЧЕТОВ")
    print(">"*60)
    
    # 1. Отчет о дрейфе данных
    drift_report = generate_data_drift_report(
        reference_data, current_data, column_mapping
    )
    
    # 2. Отчет о качестве данных
    quality_report = generate_data_quality_report(
        reference_data, current_data, column_mapping
    )
    
    # 3. Отчет о производительности модели
    performance_report = generate_model_performance_report(
        reference_data, current_data, column_mapping
    )
    
    # 4. Тесты на дрейф
    test_results = run_data_drift_tests(
        reference_data, current_data, column_mapping
    )
    
    # 5. Сравнение распределений
    distribution_comparison = compare_distributions(
        reference_data, current_data
    )
    
    # Дополнительно: тестирование с искусственным дрейфом
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ С ИСКУССТВЕННЫМ ДРЕЙФОМ")
    print("="*60)
    
    drifted_data = simulate_data_drift(current_data)
    
    drift_report_artificial = generate_data_drift_report(
        reference_data, drifted_data, column_mapping
    )
    
    report_path = "../reports/evidently_data_drift_artificial.html"
    drift_report_artificial.save_html(report_path)
    print(f"Отчет с искусственным дрейфом: {report_path}")
    
    # Итоги
    print("\n" + "="*60)
    print("МОНИТОРИНГ ЗАВЕРШЕН!")
    print("="*60)
    print("\nСгенерированные отчеты:")
    print("1. evidently_data_drift.html - Дрейф данных")
    print("2. evidently_data_quality.html - Качество данных")
    print("3. evidently_model_performance.html - Производительность модели")
    print("4. evidently_drift_tests.html - Тесты на дрейф")
    print("5. evidently_data_drift_artificial.html - Искусственный дрейф")
    print("6. distribution_comparison.csv - Сравнение распределений")
    print("\nОткройте HTML файлы в браузере для детального анализа.")


if __name__ == "__main__":
    main()
