import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#загрузка данных

print("Загрузка MNIST...")
X, y = fetch_openml('mnist_784', version=1, as_frame=False, return_X_y=True)
X = (X / 255.0) - 0.1307
y=y.astype(int)




#сплит

X_train,X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)


#параметры нейронки

input_size = 784
hidden1_size = 256
hidden2_size = 128
output_size = 10

learning_rate = 0.003
epochs = 120
batch_size = 64
dropout_rate = 0.1

np.random.seed(42)
n_train = X_train.shape[0]

#веса

W1 = np.random.randn(input_size, hidden1_size) *  np.sqrt(2. / input_size)
b1 = np.zeros((1,hidden1_size))
W2 = np.random.randn(hidden1_size,hidden2_size) * np.sqrt(2. / hidden1_size)
b2 = np.zeros((1,hidden2_size))
W3 = np.random.randn(hidden2_size,output_size) * np.sqrt(2. / hidden1_size)
b3 = np.zeros((1,output_size))

#функции

def relu(x):
    return np.maximum(0,x )

def relu_derivative(x):
    return (x>0).astype (float)

def softmax(x):
    exp_x = np.exp(x - np.max(x,axis=1,keepdims=True))
    return exp_x / np.sum(exp_x,axis=1,keepdims= True)

def cross_entropy_loss(y_pred,y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[np.arange(m), y_true])
    return np.mean(log_likelihood)


#обучение

loss_history = []
acc_history = []

print("Обучение нейронки на MNIST...")
for epoch in range(epochs):
    indices = np.random.permutation(n_train)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    epoch_loss = 0
    correct = 0 
    total = 0 


    for start in range(0, n_train,batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        batch_len = X_batch.shape[0]

        #forward pass

        z1 = X_batch @ W1 + b1
        a1 = relu(z1)
        mask1 = (np.random.rand(*a1.shape) > dropout_rate) / (1- dropout_rate)
        a1_dropped = a1 * mask1

        z2 = a1_dropped @ W2 + b2
        a2 = relu(z2)
        mask2 = (np.random.rand(*a2.shape) > dropout_rate) / (1-dropout_rate)
        a2_dropped = a2 * mask2
        
        z3 = a2_dropped @ W3 + b3
        y_pred = softmax(z3)

        loss = cross_entropy_loss(y_pred,y_batch)
        epoch_loss += loss * batch_len

        pred_labels = np.argmax(y_pred,axis=1)
        correct += np.sum(pred_labels == y_batch)
        total += batch_len

        #backward
        #градиент для выходного слоя 
        dLoss_dz3 = y_pred.copy()
        dLoss_dz3[np.arange(batch_len),y_batch] -= 1
        dLoss_dz3 /= batch_len

        dW3 = a2_dropped.T @ dLoss_dz3
        db3 = np.sum(dLoss_dz3,axis = 0,keepdims = True) 

        dLoss_da2 = dLoss_dz3 @W3.T
        dLoss_dz2 = dLoss_da2 * relu_derivative(z2) * mask2
        dW2 = a1_dropped.T @ dLoss_dz2
        db2 = np.sum(dLoss_dz2,axis=0,keepdims = True)

        dLoss_da1 = dLoss_dz2 @ W2.T
        dLoss_dz1 = dLoss_da1 * relu_derivative(z1) * mask1
        dW1 = X_batch.T @ dLoss_dz1
        db1 = np.sum(dLoss_dz1, axis=0, keepdims= True)

        #обновление весов
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1


        avg_loss = epoch_loss / n_train
        avg_acc = correct / total 
        loss_history.append(avg_loss)
        acc_history.append(avg_acc)
        print(f"Эпоха {epoch+1}/{epochs}, Loss = {avg_loss:.4f}, Train accuracy ={avg_acc:.4f}")


#тест

z1_test = X_test @ W1 + b1
a1_test = relu(z1_test) 
z2_test = a1_test @ W2 + b2
a2_test = relu(z2_test)
z3_test = a2_test @ W3 + b3
y_pred_test = softmax(z3_test)
pred_labels_test = np.argmax(y_pred_test, axis=1)
test_acc = accuracy_score(y_test,pred_labels_test)

print(f"Точность на тесте: {test_acc:.4f}")

print("Обучение логистической регрессии для сравнения...")
lr = LogisticRegression(max_iter = 1000, C = 1.0)
lr.fit(X_train, y_train) 
lr_acc = lr.score(X_test, y_test)
print(f"Точность лог.рег.: {lr_acc:.4f}")

#графики


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(loss_history)
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.title('Сходимость на MNIST')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(acc_history)
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.title('Трен точность')
plt.grid(True)
plt.show()



#примеры с предсказаиями
fig,axes = plt.subplots(2,5, figsize=(10,5))
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(X_test))
    ax.imshow(X_test[idx].reshape(28,28 ))
    ax.set_title(f'True: {y_test[idx]}, Pred: {pred_labels_test[idx]}')
    ax.axis('off')
plt.tight_layout()
plt.show()