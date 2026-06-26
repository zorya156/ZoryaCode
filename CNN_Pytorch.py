import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


#данные

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset,batch_size=64, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Тренировочных примеров: {len(train_dataset)}")
print(f"Тестовых примеров: {len(test_dataset)}")
print(f"Количество батчей в train: {len(train_loader)}")


#модель

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,8, kernel_size=3)
        self.conv2 = nn.Conv2d(8,16,kernel_size=3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*5*5, 128)
        self.fc2 = nn.Linear(128,10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16 *5 *5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    

model = CNN().to(device)

#настройка обучения

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)


#обучение

loss_history = []
acc_history = []

print("\nОбучение модели...")

for epoch in range(5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images,labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total
    loss_history.append(epoch_loss)
    acc_history.append(epoch_acc)

    print(f"Эпоха {epoch+1}/5, Loss: {epoch_loss:.4f}, train accuracy: {epoch_acc:.4f}")

#тест

model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images) 
        _, predicted = torch.max(outputs,1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = test_correct / test_total
print(f"\nТочность на тесте:{test_acc:.4f}")

#графики

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.title('Сходимость (PyTorch)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(acc_history)
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.title('Точность на тренировке')
plt.grid(True)
plt.show()

torch.save(model.state_dict(), 'mnist_cnn.pth')