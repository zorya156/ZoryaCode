#!/usr/bin/env python3
"""Простой скрипт обучения модели для классической задачи классификации раковых опухолей
Использует встроенный датасет Breast Cancer из sklearn, обучает несколько моделей и сохраняет лучшую.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def main():
    # Загрузка данных
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Разделение на трейн/тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Модели для сравнения
    models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        results[name] = (acc, model)
        print(f"Модель: {name}, Точность: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=data.target_names))

    # Выбираем лучшую модель по точности и сохраняем вместе со скейлером
    best_name, (best_acc, best_model) = max(results.items(), key=lambda kv: kv[1][0])
    print(f"Лучшая модель: {best_name} с точностью {best_acc:.4f}")
    joblib.dump({'model': best_model, 'scaler': scaler, 'feature_names': data.feature_names}, 'best_cancer_model.joblib')
    print("Модель и скейлер сохранены в best_cancer_model.joblib")


if __name__ == '__main__':
    main()
