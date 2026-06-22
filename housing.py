import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#загрузка датасета

housing = fetch_california_housing()
X = housing.data
y = housing.target

#сплит

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

#нормализация

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#линейная регрессия через градиентный спуск

np.random.seed(42)
n_features = X_train_scaled.shape[1]
w = np.random.randn(n_features) * 0.01
b = 0.0

learning_rate = 0.1
epochs = 500
n = len(X_train_scaled)

loss_history = []

print("Обучение линейной регрессии...")
for epoch in range(epochs):
    y_pred = X_train_scaled @ w + b
    #mse
    loss = np.mean((y_pred - y_train) **2)
    loss_history.append(loss)
    #градиент
    dw = (2/n) * X_train_scaled.T @ (y_pred - y_train)
    db = (2/n) * np.sum(y_pred - y_train)
    #обновление
    w -= learning_rate * dw
    b -= learning_rate * db
    if (epoch+1) % 50 == 0:
        print(f"Эпоха {epoch+1}/{epochs}, MSE = {loss:.4f}")

#оценка на тесте

y_pred_test = X_test_scaled @ w + b 
mse_manual = mean_squared_error(y_test, y_pred_test)
r2_manual = r2_score(y_test, y_pred_test)

print("\nРезультаты модели")
print (f"MSE на тесте: {mse_manual:.4f}")
print(f"R^2 на тесте: {r2_manual:.4f}")

#склеарн

lr_sklearn = LinearRegression()
lr_sklearn.fit(X_train_scaled,y_train) 
y_pred_sklearn = lr_sklearn.predict(X_test_scaled) 
mse_sklearn = mean_squared_error(y_test,y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

print("\nРезультаты склёрн")
print(f"MSE на тесте: {mse_sklearn:.4f}")
print(f"R^2 на тесте: {r2_sklearn:.4f}")


#сравнение весов

print("\nСравнение весов")
print(f"Мануал веса: {w}")
print(f"Веса склёрн: {lr_sklearn.coef_}")
print(f"Разница: {np.abs(w - lr_sklearn.coef_)}")


#графики

plt.figure(figsize=(12,5))

#график ошибки

plt.subplot(1,2,1)
plt.plot(loss_history)
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.title('Снижение MSE')
plt.grid(True)

#график предсказаний / реальных на тесте

plt.subplot(1,2,2)
plt.scatter(y_test,y_pred_test, alpha=0.3, label='Мануал предсказания',color='blue', s = 25, marker='o')
plt.scatter(y_test,y_pred_sklearn,alpha=0.3,color='red', s = 20, label='Склёрн', marker='x')
plt.plot([y_test.min(),y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Идеал', color = 'black')
plt.xlabel('Реальная цена')
plt.ylabel('Предсказанная цена')
plt.title('Предсказание / реальость')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()