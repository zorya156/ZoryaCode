import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import time



#данные

print("Загрузка данных...")
X,y = fetch_openml('mnist_784', version=1,as_frame=False,return_X_y=True)
X = (X / 255.0) - 0.1307
y = y.astype(np.int64)

#сплит

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

#преобразование векторов 28х28 + канал

def reshape_to_images(X):
    return X.reshape(-1,28,28,1)

X_train = reshape_to_images(X_train)
X_test = reshape_to_images(X_test)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

#свертка и пулинг

#свертка
def conv2d_forward(input,kernel,bias):
    """
    input: (N, H,W, C_in) 
    kernel: (Kh, Kw, C_in, C_out)
    bias(C_out,)
    returns: (N, H-Kh+1, W-Kw+1, C_out)
    """
    ###
    N, H, W, C_in = input.shape
    Kh,Kw,_,C_out = kernel.shape

    out_h = H- Kh + 1
    out_w = W - Kw+1
    output = np.zeros((N,out_h,out_w, C_out))

    for n in range(N):
        for c_out in range(C_out):
            for h in range(out_h):
                for w in range(out_w):
                    patch = input[n, h:h+Kh, w:w+Kw,:]

                    val = np.sum(patch * kernel[:,:,:, c_out]) + bias[c_out]
                    output[n,h,w,c_out] = val
    return output
#пулинг
def maxpool_forward(input, pool_size = 2, stride =2):
    """
    input: ( N, H, W, C)
    returns: (N, H//2, W//2, C)
    """
    N,H,W,C = input.shape
    out_h = H //pool_size
    out_w = W // pool_size
    output = np.zeros((N,out_h, out_w, C))
    pool_indices = np.zeros((N,out_h,out_w,C,2), dtype=np.int64)
    for n in range(N):
        for c in range(C):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    w_start = w * stride
                    patch = input[n,h_start:h_start+pool_size, w_start:w_start+pool_size,c]
                    max_val = np.max(patch)
                    max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                    output[n,h,w,c] = max_val
                    pool_indices[n,h,w,c] = [h_start + max_idx[0], w_start + max_idx[1]]
    return output, pool_indices

def relu_forward(x):
    return np.maximum(0,x)

def softmax_forward(x):
    exp_x = np.exp(x-np.max(x,axis=1,keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#веса

np.random.seed(42)

#свёрточные слои
W1 = np.random.randn(3,3,1,8) * np.sqrt(2. / (3*3*1)) 
b1 = np.zeros(8)

W2 = np.random.randn(3,3,8,16) * np.sqrt(2. / (3*3*8))
b2 = np.zeros(16)

fc1_W = np.random.randn(400, 128) * np.sqrt(2. / 400)
fc1_b = np.zeros(128)
fc2_W = np.random.randn(128, 10) * np.sqrt(2. / 128)
fc2_b = np.zeros(10)


#обучение

learning_rate = 0.01
epochs = 10
batch_size = 64
n_train = X_train.shape[0]

loss_history = []
acc_history = []

print("Начинаем обучение CNN...")
for epoch in range(epochs):
    # Перемешиваем
    perm = np.random.permutation(n_train)
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]
    total_loss = 0
    correct = 0
    
    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        batch_size_actual = end - start

        #прямой проход

        conv1 = conv2d_forward(X_batch, W1,b1) 
        relu1 = relu_forward(conv1)
        pool1, idx1 = maxpool_forward(relu1)

        conv2 = conv2d_forward(pool1, W2,b2)
        relu2 = relu_forward(conv2)
        pool2, idx2 = maxpool_forward(relu2)

        flat = pool2.reshape(batch_size_actual, - 1)

        fc1 = flat @ fc1_W + fc1_b
        relu3 = relu_forward(fc1)

        fc2 = relu3 @fc2_W + fc2_b 
        probs = softmax_forward(fc2) 
        
        #loss, accuracy

        loss = -np.log(probs[np.arange(batch_size_actual), y_batch] + 1e-9).mean()
        total_loss += loss * batch_size_actual
        preds = np.argmax(probs, axis = 1)
        correct += np.sum(preds == y_batch)

        #forward pass

        d_fc2 = probs.copy()
        d_fc2[np.arange(batch_size_actual), y_batch] -= 1
        d_fc2 /= batch_size_actual

        dW_fc2 = relu3.T @ d_fc2
        db_fc2 = np.sum(d_fc2, axis=0)   # -> (10,), а не (1,10)
        
        d_relu3 = d_fc2 @ fc2_W.T
        d_fc1 = d_relu3 * (fc1 > 0).astype(float)

        dW_fc1 = flat.T @ d_fc1
        db_fc1 = np.sum(d_fc1, axis=0)   # -> (128,)

        d_flat = d_fc1 @ fc1_W.T
        d_pool2 = d_flat.reshape(pool2.shape)

        d_relu2 = np.zeros_like(relu2)
        for n in range(batch_size_actual):
            for c in range(16):
                for h in range(5):
                    for w in range (5): 
                        h_orig, w_orig = idx2[n,h,w,c]
                        d_relu2[n,h_orig, w_orig, c] += d_pool2[n,h,w,c]
        d_conv2 = d_relu2 * (conv2 > 0).astype(float)

        dW2 = np.zeros_like(W2)
        db2 = np.sum(d_conv2, axis=(0,1,2))   # -> (16,)

        for n in range(batch_size_actual):
            for c_out in range(16):
                for h in range(11):
                    for w in range(11):
                        patch = pool1[n, h:h+3, w:w+3, :]  # (3,3,8)
                        dW2[:, :, :, c_out] += patch * d_conv2[n, h, w, c_out]

        d_pool1 = np.zeros_like(pool1)
        for n in range(batch_size_actual):
            for c_out in range(16):
                for h in range(11):
                    for w in range(11):
                        for kh in range(3):
                            for kw in range(3):
                                d_pool1[n, h+kh, w+kw, :] += W2[kh, kw, :, c_out] * d_conv2[n, h, w, c_out]
       
        d_relu1 = np.zeros_like(relu1)
        for n in range(batch_size_actual):
            for c in range(8):
                for h in range(13):
                    for w in range(13):
                        h_orig, w_orig = idx1[n, h, w, c]
                        d_relu1[n, h_orig, w_orig, c] += d_pool1[n, h, w, c]

        d_conv1 = d_relu1 * (conv1 > 0).astype(float)

        dW1 = np.zeros_like(W1)
        db1 = np.sum(d_conv1, axis=(0,1,2))   # -> (8,)
        for n in range(batch_size_actual):
            for c_out in range(8):
                for h in range(26):
                    for w in range(26):
                        patch = X_batch[n, h:h+3, w:w+3, :]  # (3,3,1)
                        dW1[:, :, :, c_out] += patch * d_conv1[n, h, w, c_out]

        #обновление весов
        fc2_W -= learning_rate * dW_fc2
        fc2_b -= learning_rate * db_fc2
        fc1_W -= learning_rate * dW_fc1
        fc1_b -= learning_rate * db_fc1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    avg_loss = total_loss / n_train
    acc = correct / n_train
    loss_history.append(avg_loss)
    acc_history.append(acc)
    print(f"Эпоха {epoch+1}/{epochs}, Loss = {avg_loss:.4f}, Train Accuracy = {acc:.4f}")


#тест

def predict(X):
    N = X.shape[0]
    conv1 = conv2d_forward(X,W1,b1) 
    relu1 = relu_forward(conv1)
    pool1, _ = maxpool_forward(relu1)
    conv2 - conv2d_forward(pool1, W2,b2)
    relu2 = relu_forward(conv2)
    pool2, _ = maxpool_forward(relu2)
    flat = pool2.reshape(N, -1)
    fc1 = flat @ fc1_W + fc1_b
    relu3 = relu_forward(fc1)
    fc2 = relu3 @ fc2_W + fc2_b
    probs = softmax_forward(fc2)
    return np.argmax(probs, axis=1)

y_pred = predict(X_test)
test_acc = accuracy_score(y_test,y_pred)
print(f"\nТочность на тесте: {test_acc:.4f}")

#логРег

print("Обучаем логистическую регрессию для сравнения...")
lr = LogisticRegression(max_iter=1000, C=1.0)
X_train_flat = X_train.reshape(-1, 784)
X_test_flat = X_test.reshape(-1, 784)
lr.fit(X_train_flat, y_train)
lr_acc = lr.score(X_test_flat, y_test)
print(f"Точность лог.рег.: {lr_acc:.4f}")


#графики
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.title('Сходимость CNN')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(acc_history)
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.title('Точность на тренировке')
plt.grid(True)
plt.show()





