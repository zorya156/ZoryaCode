import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import math
from sklearn.preprocessing import StandardScaler



df = sns.load_dataset('titanic')
df_numeric = df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']].copy() 
df_numeric['sex_code'] = (df['sex'] == 'male').astype(int)
df['age'] = df.groupby('pclass')['age'].transform(lambda x: x.fillna(x.median()))


X = df[['pclass','age','sex']].copy()
X['sex'] == (X['sex'] == 'female').astype(int)
X = X.drop('sex', axis=1)


y = df['survived']
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

np.random.seed(42)
w = np.random.randn(X_train_scaled.shape[1]) * 0.01
b= 0.0

learning_rate = 0.1
epochs = 1000
n=len(X_train_scaled)

history_w=[]
history_loss=[]

print("Обучение модели...")

for epoch in range(epochs):
    z = X_train_scaled @ w + b
    y_pred = 1 / (1 + np.exp(-z)) 
    loss = - (y_train * np.log(y_pred - 1e-9) + (1-y_train)*np.log(1-y_pred+1e-9)).mean()

    dw=(X_train_scaled.T @ (y_pred - y_train)) /n 
    db = (y_pred - y_train ).mean()

    w=w-learning_rate * dw
    b = b - learning_rate * db


    if epoch % 10 == 0:
        history_w.append(w.copy())
        history_loss.append(loss)

    if epoch % 100 ==0:
        print(f"Эпоха {epoch}: Loss = {loss:.4f}")


z_test = X_test_scaled @ w + b

y_pred_test = 1 / (1+np.exp((-z_test))) >= 0.5
accuracy = (y_pred_test == y_test).mean()
print(f"\n Точность модели: {accuracy:.2%}")

sk_model = LogisticRegression(max_iter=1000)
sk_model.fit(X_train_scaled, y_train)
sk_acc = sk_model.score(X_test_scaled, y_test)
print(f"Точность Sklearn: {sk_acc:.2%}\n")

history_w = np.array(history_w)  # (кол-во точек, 2)

min_w1, max_w1 = history_w[:, 0].min(), history_w[:, 0].max()
min_w2, max_w2 = history_w[:, 1].min(), history_w[:, 1].max()


padding1 = max(0.5, (max_w1 - min_w1) * 0.3)
padding2 = max(0.5, (max_w2 - min_w2) * 0.3)

w1_range = np.linspace(min_w1 - padding1, max_w1 + padding1, 50)
w2_range = np.linspace(min_w2 - padding2, max_w2 + padding2, 50)

print(f"Диапазон w1: [{w1_range.min():.2f}, {w1_range.max():.2f}]")
print(f"Диапазон w2: [{w2_range.min():.2f}, {w2_range.max():.2f}]")

W1, W2 = np.meshgrid(w1_range, w2_range)


loss_grid = np.zeros_like(W1)
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w_tmp = np.array([W1[i, j], W2[i, j]])
        z_tmp = X_train_scaled @ w_tmp + b  # b фиксируем (итоговое)
        y_pred_tmp = 1 / (1 + np.exp(-z_tmp))
        loss_tmp = - (y_train * np.log(y_pred_tmp + 1e-9) + (1 - y_train) * np.log(1 - y_pred_tmp + 1e-9)).mean()
        loss_grid[i, j] = loss_tmp


fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(W1, W2, loss_grid, cmap='viridis', alpha=0.7, edgecolor='none')


z_history = np.array(history_loss) + 0.005
ax1.plot(history_w[:, 0], history_w[:, 1], z_history, 
         color='red', linewidth=2, marker='o', markersize=3, label='Траектория')
ax1.scatter(history_w[0, 0], history_w[0, 1], z_history[0], 
            color='green', s=80, label='Старт')
ax1.scatter(history_w[-1, 0], history_w[-1, 1], z_history[-1], 
            color='red', s=80, label='Финиш')
ax1.set_xlabel('Вес для класса (pclass)')
ax1.set_ylabel('Вес для пола (sex)')
ax1.set_zlabel('Log Loss')
ax1.set_title('3D поверхность ошибки и траектория')
ax1.legend()


ax2 = fig.add_subplot(122)
contour = ax2.contour(W1, W2, loss_grid, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.plot(history_w[:, 0], history_w[:, 1], 'r-', linewidth=2, label='Траектория')
ax2.scatter(history_w[0, 0], history_w[0, 1], color='green', s=80, label='Старт')
ax2.scatter(history_w[-1, 0], history_w[-1, 1], color='red', s=80, label='Финиш')
# Точка Sklearn
ax2.scatter(sk_model.coef_[0][0], sk_model.coef_[0][1], 
            color='blue', s=100, marker='*', label='Sklearn')
ax2.set_xlabel('Вес для класса (pclass)')
ax2.set_ylabel('Вес для пола (sex)')
ax2.set_title('Контурная карта ошибки (вид сверху)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"Наши финальные веса: {w}")
print(f"Веса от Sklearn: {sk_model.coef_[0]}")
print(f"Разница: {np.abs(w - sk_model.coef_[0])}")






















































# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# train_accuracy = model.score(X_train, y_train)
# test_accuracy = model.score(X_test, y_test)

# print(f"Тренировочная аккураси: {train_accuracy:.2f}")
# print(f"Тестовая аккураси: {test_accuracy:.2f}")


# probs = model.predict_proba(X)[:, 1]  # Теперь длина probs = 891


# plt.figure(figsize=(10, 6))


# plt.scatter(X['pclass'], y, color='blue', alpha=0.4, label='Реальные данные (0/1)')


# plt.scatter(X['age'], probs, color='green', alpha=0.3, label='Вероятность выживаия')

# plt.xlabel('Возраст')
# plt.ylabel('Вероятность выжить')
# plt.title('Зависимость вероятности от возраста')
# plt.grid(True)
# plt.legend()

# plt.show()



# print(f"Количество признаков в X: {X.shape[1]}")  # Должно быть 3 (pclass, age, sex)
# print(f"Количество весов у модели: {len(model.coef_[0])}")  # Должно совпадать с X.shape[1]


# weights = model.coef_[0]  
# bias = model.intercept_[0]

# print(f"Веса модели: {weights}")
# print(f"Смещение (bias): {bias}")


# def manual_predict(row, weights, bias):
    
#     raw_score = 0
#     for i in range(len(weights)):
#         raw_score += row.iloc[i] * weights[i]  # iloc[i] берет i-й по счету признак
#     raw_score += bias
#     return raw_score


# first_row = X.iloc[0]
# raw = manual_predict(first_row, weights, bias)



# prob_manual = 1 / (1 + math.exp(-raw))


# prob_sklearn = model.predict_proba(X.iloc[[0]])[0][1]

# print(f"Сырой score (руками): {raw:.4f}")
# print(f"Вероятность (руками): {prob_manual:.4%}")
# print(f"Вероятность (Sklearn): {prob_sklearn:.4%}")


# if abs(prob_manual - prob_sklearn) < 0.0001:
#     print("Работает")
# else:
#     print("Не работает")





# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler  # Пригодится

# # 1. Данные (берем только 2 признака для простоты)
# df = sns.load_dataset('titanic')
# X = df[['pclass', 'sex']].copy()
# X['sex'] = (X['sex'] == 'female').astype(int)
# y = df['survived'].values  # Берем как numpy массив

# # Разделяем
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # [ВАЖНО] Нормализация! Для градиентного спуска это критично.
# # Класс (1,2,3) и пол (0,1) в одном масштабе - ок, но на будущее запомни.
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Добавляем столбец единиц для удобства (чтобы не таскать b отдельно), но мы сделаем через b.

# # ============================================
# # РЕАЛИЗУЕМ ГРАДИЕНТНЫЙ СПУСК С НУЛЯ
# # ============================================

# # Инициализация весов и смещения
# np.random.seed(42)  # Чтобы результат был повторяемым
# w = np.random.randn(X_train_scaled.shape[1]) * 0.01
# b = 0.0

# learning_rate = 0.1
# epochs = 1000
# n = len(X_train_scaled)

# # Массивы для записи истории ошибки
# loss_history = []

# print("Начинаем обучение...")
# for epoch in range(epochs):
#     # 1. Прямой проход (Forward pass)
#     z = X_train_scaled @ w + b
#     y_pred = 1 / (1 + np.exp(-z))  # Сигмоида
    
#     # 2. Вычисляем ошибку (Log Loss)
#     loss = - (y_train * np.log(y_pred + 1e-9) + (1 - y_train) * np.log(1 - y_pred + 1e-9)).mean()
#     loss_history.append(loss)
    
#     # 3. Вычисляем градиенты (Самая важная магия!)
#     dw = (X_train_scaled.T @ (y_pred - y_train)) / n
#     db = (y_pred - y_train).mean()
    
#     # 4. Обновляем веса (Делаем шаг в сторону уменьшения ошибки)
#     w = w - learning_rate * dw
#     b = b - learning_rate * db
    
#     # Выводим прогресс каждые 100 шагов
#     if epoch % 100 == 0:
#         print(f"Эпоха {epoch}: Loss = {loss:.4f}")

# # ============================================
# # ПРОВЕРКА НА ТЕСТЕ
# # ============================================

# # Предсказания на тестовой выборке
# z_test = X_test_scaled @ w + b
# y_pred_test = (1 / (1 + np.exp(-z_test))) >= 0.5  # Превращаем в 0 или 1

# # Считаем точность руками
# accuracy = (y_pred_test == y_test).mean()
# print(f"\n✅ Точность модели (наша, с нуля): {accuracy:.2%}")

# # Сравним с Sklearn (для калибровки)
# from sklearn.linear_model import LogisticRegression
# sk_model = LogisticRegression(max_iter=1000)
# sk_model.fit(X_train_scaled, y_train)
# sk_acc = sk_model.score(X_test_scaled, y_test)
# print(f"Точность модели Sklearn: {sk_acc:.2%}")

# # ============================================
# # ГРАФИК СХОДИМОСТИ (как качаются веса)
# # ============================================

# plt.figure(figsize=(12, 4))

# # График 1: Падение ошибки (Loss)
# plt.subplot(1, 2, 1)
# plt.plot(loss_history)
# plt.xlabel('Эпоха обучения')
# plt.ylabel('Значение ошибки (Log Loss)')
# plt.title('Как ошибка уменьшается со временем')
# plt.grid(True)

# # График 2: Сравнение весов (наши против Sklearn)
# plt.subplot(1, 2, 2)
# plt.bar(['Вес для класса', 'Вес для пола'], 
#         [w[0], sk_model.coef_[0][0]], alpha=0.6, label='Наши')
# plt.bar(['Вес для класса', 'Вес для пола'], 
#         [sk_model.coef_[0][0], sk_model.coef_[0][1]], alpha=0.6, label='Sklearn')
# plt.title('Сравнение весов')
# plt.legend()
# plt.grid(True)

# plt.show()

# print("\nНаши финальные веса:", w)
# print("Веса от Sklearn:", sk_model.coef_[0])