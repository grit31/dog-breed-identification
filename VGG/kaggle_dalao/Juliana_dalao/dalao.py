import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim import SGD

# ==== 路径配置 ====
data_dir = '../input/dog-breed-identification'
save_dir = './'

# ==== 数据整理（与 Keras 版本一致）====
labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
breeds = labels.breed.unique()
n_class = len(breeds)
print('类别数:', n_class)

# 构建类别到编号的映射
breed2idx = {breed: idx for idx, breed in enumerate(breeds)}
labels['breed_idx'] = labels['breed'].map(breed2idx)

# ==== 数据增强与 Dataset ====
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

class DogBreedDataset(Dataset):
    def __init__(self, df, root, transform=None):
        self.df = df
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, f"{row['id']}.jpg")
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row['breed_idx'])
        return img, label

# ==== 划分训练/验证 ====
train_df, val_df = train_test_split(labels, test_size=0.2, stratify=labels['breed_idx'], random_state=42)
train_ds = DogBreedDataset(train_df, os.path.join(data_dir, 'train'), transform=transform_train)
val_ds = DogBreedDataset(val_df, os.path.join(data_dir, 'train'), transform=transform_val)
batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

# ==== 手动实现 VGG11（可动态配置）====
class VGG(nn.Module):
    def __init__(self, cfg, num_classes=120):
        super().__init__()
        self.features = self._make_layers(cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# VGG11结构
cfg_vgg11 = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

# ==== 设备配置 ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG(cfg_vgg11, num_classes=n_class).to(device)

# ==== 优化器和损失 ====
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# ==== 训练主循环 ====
n_epochs = 5
for epoch in range(1, n_epochs+1):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    acc = correct / total
    print(f"Epoch {epoch} | Train loss: {train_loss/total:.4f} | Acc: {acc:.4f}")

    # 验证
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    val_acc = val_correct / val_total
    print(f"Epoch {epoch} | Val loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.4f}")

# ==== 测试集推理并生成提交文件 ====
class DogTestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.img_names = sorted(os.listdir(test_dir))
        self.test_dir = test_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.test_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        img_id = os.path.splitext(img_name)[0]
        return img, img_id

test_dir = os.path.join(data_dir, 'test')
test_ds = DogTestDataset(test_dir, transform=transform_val)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

model.eval()
probs = []
img_ids = []
with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        pred_probs = torch.softmax(outputs, dim=1).cpu().numpy()
        probs.append(pred_probs)
        img_ids.extend(ids)
probs = np.vstack(probs)

# 保存提交
columns = list(breeds)
rows = []
for i, img_id in enumerate(img_ids):
    pred = ["%.4f" % p for p in probs[i]]
    rows.append([img_id] + pred)
columns = ["id"] + columns

save_path = os.path.join(save_dir, 'submission.csv')
import csv
with open(save_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(rows)
print(f"提交文件已保存到 {save_path}")
