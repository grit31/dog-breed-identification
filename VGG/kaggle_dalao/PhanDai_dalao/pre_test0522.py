import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===== 路径配置 =====
data_dir = r'D:\homework\Course_Design_of_artificial_intelligence\third\dog-breed-identification'
save_dir = r'D:\homework\Course_Design_of_artificial_intelligence\third\VGG\kaggle_dalao\PhanDai_dalao'

# # ===== VGG11网络定义（完全兼容torchvision官方预训练权重） =====
# cfg_vgg11 = [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M']


# def make_layers(cfg, in_channels=3, batch_norm=False):
#     layers = []
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)

# class VGG11(nn.Module):
#     def __init__(self, num_classes=120, batch_norm=False):
#         super().__init__()
#         self.features = make_layers(cfg_vgg11, batch_norm=batch_norm)
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(0.5),
#             nn.Linear(4096, num_classes),
#         )
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

# ---- 修正cfg ----
cfg_vgg11 = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG11(nn.Module):
    def __init__(self, num_classes=120, batch_norm=False):
        super().__init__()
        self.features = make_layers(cfg_vgg11, batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ====== 加载标签 =======
labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
labels['image_path'] = labels['id'].apply(lambda x: os.path.join(data_dir, 'train', f'{x}.jpg'))
encoder = LabelEncoder()
labels['breed'] = encoder.fit_transform(labels['breed'])

# ====== 数据划分 =======
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    labels, test_size=0.2, random_state=42, stratify=labels['breed'])

# ====== 数据集类 =======
class DogBreedDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        label = self.df.loc[idx, 'breed']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.2),
    transforms.RandomVerticalFlip(0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = DogBreedDataset(train_df, transform=train_transform)
val_dataset = DogBreedDataset(val_df, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# # ====== 模型初始化及加载预训练权重 =======
# import torchvision
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Step1：构建1000类模型加载官方权重
# model = VGG11(num_classes=1000, batch_norm=False)
# state_dict = torchvision.models.vgg11(pretrained=True).state_dict()
# model.load_state_dict(state_dict)
# # Step2：替换最后fc层，改为本任务类别数（如120）
# model.classifier[-1] = nn.Linear(4096, len(encoder.classes_))
# model = model.to(device)

import torchvision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG11(num_classes=1000, batch_norm=False)
# 加载官方预训练
state_dict = torchvision.models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1').state_dict()
model.load_state_dict(state_dict)  # 不会报错
model.classifier[-1] = nn.Linear(4096, 120)
model = model.to(device)

# ====== 损失函数和优化器 =======
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00005)#最开始0.0001
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# ====== 训练主循环 =======
train_losses, val_losses = [], []
epochs = 30
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for imgs, labels_ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - train"):
        imgs = imgs.to(device)
        labels_ = labels_.to(device).long()
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels_)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # 验证
    model.eval()
    val_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels_ in tqdm(val_loader, desc="val"):
            imgs = imgs.to(device)
            labels_ = labels_.to(device).long()
            output = model(imgs)
            loss = criterion(output, labels_)
            val_loss += loss.item() * imgs.size(0)
            preds = output.argmax(1)
            correct += (preds == labels_).sum().item()
            total += labels_.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    acc = correct / total
    scheduler.step()
    print(f'Epoch {epoch+1}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, val acc {acc:.4f}')
    torch.save(model.state_dict(), os.path.join(save_dir, f'vgg11_epoch{epoch+1}.pth'))

plt.figure(figsize=(10,6))
plt.plot(range(1, epochs+1), train_losses, label='train loss')
plt.plot(range(1, epochs+1), val_losses, label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training/Validation Loss')
plt.grid(True)
plt.savefig('Training_Validation Loss')
plt.show()

# ====== 推理与生成Kaggle提交文件 =======
class Dog_Breed_Dataset_test(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transforms = transform
        self.image_names = [f.split('.')[0] for f in os.listdir(root_dir) if f.endswith('.jpg')]
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name + '.jpg')
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, img_name

test_dir = os.path.join(data_dir, 'test')
test_dataset = Dog_Breed_Dataset_test(root_dir=test_dir, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
all_predictions = []
dog_breeds = encoder.classes_
with torch.no_grad():
    for imgs, img_names in tqdm(test_loader, desc='predict'):
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        for i, img_name in enumerate(img_names):
            for class_idx, prob in enumerate(probs[i]):
                class_name = dog_breeds[class_idx]
                all_predictions.append({'class_name': class_name, 'probability': prob, 'image_name': img_name})

all_predictions_df = pd.DataFrame.from_records(all_predictions)
submissions = all_predictions_df.pivot(index='image_name', columns='class_name', values='probability').reset_index()
submissions = submissions.rename(columns={'image_name': 'id'})
submissions.to_csv(os.path.join(save_dir, 'submission.csv'), index=False)
print('提交文件已保存:', os.path.join(save_dir, 'submission.csv'))

