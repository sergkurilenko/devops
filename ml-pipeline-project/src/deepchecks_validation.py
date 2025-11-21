"""
Проверка качества данных с использованием Deepchecks
"""
import os
import warnings

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, model_evaluation
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def load_data():
    """Загрузка данных"""
    print("Загрузка данных California Housing...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    return X, y


def create_deepchecks_datasets(X, y, test_size=0.2):
    """Создание датасетов для Deepchecks"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Создание DataFrame с целевой переменной
    train_df = X_train.copy()
    train_df['MedHouseVal'] = y_train
    
    test_df = X_test.copy()
    test_df['MedHouseVal'] = y_test
    
    # Создание Deepchecks Dataset
    ds_train = Dataset(
        train_df,
        label='MedHouseVal',
        cat_features=[]
    )
    
    ds_test = Dataset(
        test_df,
        label='MedHouseVal',
        cat_features=[]
    )
    
    return ds_train, ds_test


def run_data_integrity_checks(ds_train, ds_test):
    """Запуск проверок целостности данных"""
    print("\n" + "="*60)
    print("ПРОВЕРКА ЦЕЛОСТНОСТИ ДАННЫХ (Data Integrity)")
    print("="*60)
    
    suite = data_integrity()
    result = suite.run(train_dataset=ds_train, test_dataset=ds_test)
    
    # Сохранение отчета
    os.makedirs("../reports", exist_ok=True)
    report_path = "../reports/deepchecks_data_integrity.html"
    result.save_as_html(report_path)
    
    print(f"\nОтчет сохранен: {report_path}")
    print("\nРезультаты проверок:")
    print(result)
    
    return result


def run_model_evaluation(ds_train, ds_test, model):
    """Запуск проверок оценки модели"""
    print("\n" + "="*60)
    print("ОЦЕНКА МОДЕЛИ (Model Evaluation)")
    print("="*60)
    
    suite = model_evaluation()
    result = suite.run(
        train_dataset=ds_train,
        test_dataset=ds_test,
        model=model
    )
    
    # Сохранение отчета
    report_path = "../reports/deepchecks_model_evaluation.html"
    result.save_as_html(report_path)
    
    print(f"\nОтчет сохранен: {report_path}")
    print("\nРезультаты проверок:")
    print(result)
    
    return result


def run_custom_checks(ds_train, ds_test):
    """Запуск пользовательских проверок"""
    print("\n" + "="*60)
    print("ПОЛЬЗОВАТЕЛЬСКИЕ ПРОВЕРКИ")
    print("="*60)
    
    from deepchecks.tabular.checks import (
        DataDuplicates,
        MixedDataTypes,
        MixedNulls,
        StringMismatch,
        IsSingleValue,
        SpecialCharacters,
        FeatureLabelCorrelation
    )
    
    # Список проверок
    checks = [
        DataDuplicates(),
        MixedDataTypes(),
        MixedNulls(),
        StringMismatch(),
        IsSingleValue(),
        SpecialCharacters(),
        FeatureLabelCorrelation()
    ]
    
    results = []
    for check in checks:
        print(f"\nВыполнение: {check.name()}")
        try:
            result = check.run(dataset=ds_train)
            results.append({
                'check': check.name(),
                'status': 'passed' if result.passed_conditions() else 'failed',
                'result': result
            })
        except Exception as e:
            print(f"Ошибка: {e}")
            results.append({
                'check': check.name(),
                'status': 'error',
                'result': str(e)
            })
    
    # Вывод сводки
    print("\n" + "-"*60)
    print("СВОДКА ПРОВЕРОК:")
    print("-"*60)
    for res in results:
        status_symbol = "✓" if res['status'] == 'passed' else "✗"
        print(f"{status_symbol} {res['check']}: {res['status']}")
    
    return results


def main():
    """Основная функция"""
    print("="*60)
    print("DEEPCHECKS: ПРОВЕРКА КАЧЕСТВА ДАННЫХ И МОДЕЛИ")
    print("="*60)
    
    # Загрузка данных
    X, y = load_data()
    print(f"Размер данных: {X.shape}")
    
    # Создание датасетов
    ds_train, ds_test = create_deepchecks_datasets(X, y)
    
    # Проверка целостности данных
    integrity_results = run_data_integrity_checks(ds_train, ds_test)
    
    # Обучение простой модели для проверок
    print("\nОбучение модели для проверок...")
    X_train = ds_train.data.drop('MedHouseVal', axis=1)
    y_train = ds_train.data['MedHouseVal']
    
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    print("Модель обучена!")
    
    # Оценка модели
    evaluation_results = run_model_evaluation(ds_train, ds_test, model)
    
    # Пользовательские проверки
    custom_results = run_custom_checks(ds_train, ds_test)
    
    print("\n" + "="*60)
    print("ПРОВЕРКА ЗАВЕРШЕНА!")
    print("="*60)
    print("\nОтчеты сохранены в директории: reports/")
    print("- deepchecks_data_integrity.html")
    print("- deepchecks_model_evaluation.html")
    print("\nОткройте HTML файлы в браузере для детального анализа.")


if __name__ == "__main__":
    main()
