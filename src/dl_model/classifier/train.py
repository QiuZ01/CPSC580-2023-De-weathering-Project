import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from model import shufflenet_v2_x2_0

dataset_path = '../../data/classify'

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 或其他大小
    transforms.CenterCrop(224),     # 根据 ShuffleNet 的输入调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = ImageFolder(root=dataset_path, transform=transform)

# 数据集分割
total_size = len(full_dataset)
train_size = int(0.7 * total_size)  # 训练集占70%
test_size = int(0.15 * total_size)  # 测试集占15%
val_size = total_size - train_size - test_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

model = shufflenet_v2_x2_0()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

log_file = './training_log.txt'
with open(log_file, 'w') as logf:
    logf.write('Training Log\n')

# train & val
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # val
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    best_acc = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    log_msg = f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {100 * correct / total}%\n'
    print(log_msg)
    with open(log_file, 'a') as logf:
        logf.write(log_msg)

    if (100 * correct / total > best_acc):
        best_acc = 100 * correct / total
        torch.save(model.state_dict(), f'best.pth')

torch.save(model.state_dict(), f'last.pth')



# test
model = shufflenet_v2_x2_0()
model.load_state_dict(torch.load('./best.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Loss: {test_loss/len(test_loader)}, Test Accuracy: {100 * correct / total}%')