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
hidden_size = 16
output_size = 1

learning_rate = 0.01
epochs= 1000
batch_size = 64
n_train = X_train_scaled.shape[0]

#веса

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
b1 = np.zeros((1,hidden_size))
W2 = np.random.randn(hidden_size,output_size) * np.sqrt(2. / hidden_size)
b2 = np.zeros((1, output_size))


#relu и производные

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x>0).astype(float)


#обучение

loss_history = []

print("Обучение MLP...")
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

        #форвард пасс
        #скрытый слой
        z1 = X_batch @ W1 + b1 
        a1 = relu(z1)
        #выходной слой
        z2 = a1 @ W2 + b2
        y_pred = z2
        #mse batch
        loss = np.mean((y_pred - y_batch) ** 2)
        epoch_loss += loss * batch_len

        #backpropagation

        #градиент на входе
        dLoss_dy = (2/batch_len) * (y_pred - y_batch)
        #градиенты w2, b2
        dLoss_dz2 = dLoss_dy
        dW2 = (a1.T @ dLoss_dz2)
        db2 = np.sum(dLoss_dz2, axis = 0, keepdims = True)
        #градиент скрытого слоя
        dLoss_da1 = dLoss_dz2 @ W2.T
        dLoss_dz1 = dLoss_da1 * relu_derivative(z1)
        dW1 = (X_batch.T @ dLoss_dz1)
        db1 = np.sum(dLoss_dz1,axis = 0, keepdims = True)

        #обновление весов

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
y_pred_test = a1_test @ W2 + b2

mse_manual = mean_squared_error(y_test, y_pred_test)
r2_manual = r2_score(y_test,y_pred_test)

print("\nРезультаты нейросети")
print(f"MSE на тесте: {mse_manual:.4f}")
print(f"R^2 на тесте: {r2_manual:.4f}")


#линейная регрессия склёрн

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\n Линейная регрессия для сравнения")
print(f"MSE на тесте(LR): {mse_lr:.4f}")
print(f"R^2 на тесте(LR): {r2_lr:.4f}")



#графики

plt.figure(figsize = (12,5))

plt.subplot(1,2,1)
plt.plot(loss_history)
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.title("Снижение ошибки нейросети")
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_test, alpha=0.5,color='blue', label = 'Нейросеть')
plt.scatter(y_test,y_pred_lr, alpha=0.4, color = 'red', label= 'Линейная регрессия')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label= 'Идеал')
plt.xlabel('Реальная цена')
plt.ylabel('Предсказанная цена')
plt.title('Предсказание нейронки / линейной регрессии')
plt.legend()
plt.grid(True)

plt.tight_layout
plt.show()
