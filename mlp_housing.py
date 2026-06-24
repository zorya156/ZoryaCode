import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


#подготовка данных

data = fetch_california_housing()

X = data.data
y = data.target.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)

#параметры нейронки

input_size = X_train_scaled.shape[1]
hidden1_size = 32
hidden2_size = 16
output_size = 1

learning_rate = 0.001
epochs= 3900
batch_size = 64
dropout_rate = 0.2

np.random.seed(42)
n_train = X_train_scaled.shape[0]

#веса


W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2. / input_size)
b1 = np.zeros((1, hidden1_size))

W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2. / hidden1_size)
b2 = np.zeros((1, hidden2_size))

W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2. / hidden2_size)
b3 = np.zeros((1, output_size))

#relu и производные

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x>0).astype(float)


#обучение

loss_history = []

print("Обучение глубокой нейросети")
for epoch in range(epochs):
    #перемешивание данных
    indices = np.random.permutation(n_train)
    X_shuffled = X_train_scaled[indices]
    y_shuffled = y_train[indices]

    epoch_loss = 0


    
    for start in range(0, n_train, batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        batch_len = X_batch.shape[0]
        
        
        # Слой 1
        z1 = X_batch @ W1 + b1          # (batch, 32)
        a1 = relu(z1)                   # (batch, 32)
        
        # Dropout на слое 1 (во время обучения)
        mask1 = (np.random.rand(*a1.shape) > dropout_rate) / (1 - dropout_rate)
        a1_dropped = a1 * mask1         # (batch, 32)
        
        # Слой 2
        z2 = a1_dropped @ W2 + b2       # (batch, 16)
        a2 = relu(z2)                   # (batch, 16)
        
        # Dropout на слое 2
        mask2 = (np.random.rand(*a2.shape) > dropout_rate) / (1 - dropout_rate)
        a2_dropped = a2 * mask2         # (batch, 16)
        
        # Выходной слой (линейный)
        z3 = a2_dropped @ W3 + b3       # (batch, 1)
        y_pred = z3
        
        # Ошибка (MSE)
        loss = np.mean((y_pred - y_batch) ** 2)
        epoch_loss += loss * batch_len
        
        #backpropagation

         # Градиент на выходе
        dLoss_dz3 = (2 / batch_len) * (y_pred - y_batch)  # (batch, 1)
        
        # Градиенты для W3 и b3
        dW3 = a2_dropped.T @ dLoss_dz3
        db3 = np.sum(dLoss_dz3, axis=0, keepdims=True)
        
        # Градиент до слоя 2 
        dLoss_da2 = dLoss_dz3 @ W3.T
        dLoss_dz2 = dLoss_da2 * relu_derivative(z2) * mask2  # умножаем на маску Dropout
        
        dW2 = a1_dropped.T @ dLoss_dz2
        db2 = np.sum(dLoss_dz2, axis=0, keepdims=True)
        
        # Градиент до слоя 1 
        dLoss_da1 = dLoss_dz2 @ W2.T
        dLoss_dz1 = dLoss_da1 * relu_derivative(z1) * mask1
        
        dW1 = X_batch.T @ dLoss_dz1
        db1 = np.sum(dLoss_dz1, axis=0, keepdims=True)


        #обновление весов

        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    avg_loss = epoch_loss / n_train 
    loss_history.append(avg_loss)

    if (epoch+1) % 100 == 0:
        print(f"Эпоха {epoch+1}/{epochs}, MSE = {avg_loss:.4f}")
        
#оценка на тесте

#прямой проход на тесте
z1_test = X_test_scaled @ W1 + b1
a1_test = relu(z1_test)
z2_test = a1_test @ W2 + b2
a2_test = relu(z2_test)
y_pred_test = a2_test @ W3 + b3

mse_deep = mean_squared_error(y_test, y_pred_test)
r2_deep = r2_score(y_test, y_pred_test)


print("\nРезультаты глубокой нейросети")
print(f"MSE на тесте: {mse_deep:.4f}")
print(f"R^2 на тесте: {r2_deep:.4f}")



#линейная регрессия склёрн

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nДля сравнения:")
print(f"Линейная регрессия R^2: {r2_lr:.4f}")
print(f"Однослойная MLP R² (old): 0.7572")
print(f"Глубокая MLP R² (new): {r2_deep:.4f}")





#графики

plt.figure(figsize = (12,5))

plt.subplot(1,2,1)
plt.plot(loss_history)
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.title("Сходимость глубокой нейросети")
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_test, alpha=0.5,color='blue', label = 'Глубокая нейросеть')
plt.scatter(y_test,y_pred_lr, alpha=0.4, color = 'red', label= 'Линейная регрессия')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label= 'Идеал')
plt.xlabel('Реальная цена')
plt.ylabel('Предсказанная цена')
plt.title('Предсказание нейронки / линейной регрессии')
plt.legend()
plt.grid(True)

plt.tight_layout
plt.show()
