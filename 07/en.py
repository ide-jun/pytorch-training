import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import cifer_datasets
from model import CNN

train_data, test_data = cifer_datasets()
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data , batch_size=64, shuffle=False)

model = CNN()

criterion = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

epochs = 20

for epoch in range(epochs):
    train_loss = 0
    val_loss   = 0
    val_acc    = 0

    model.train()
    for images, labels in train_loader:
        # 勾配を初期化
        optimizer.zero_grad()

        train_outputs = model(images)
        loss          = criterion(train_outputs, labels)
        train_loss   += loss.item()

        # 誤差逆伝播
        loss.backward()

        # 重みを更新
        optimizer.step()
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            test_outputs = model(images)
            loss         = criterion(test_outputs, labels)
            val_loss    += loss.item()
            val_acc     += (test_outputs.max(1)[1] == labels).sum().item()
            print(val_acc)
        avg_val_loss = val_loss / len(test_loader)
        avg_val_acc  = val_acc  / len(test_loader.dataset)
    print(f"Epoch {epoch+1}, train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}")