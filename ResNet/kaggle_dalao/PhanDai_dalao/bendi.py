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
save_dir = r'D:\homework\Course_Design_of_artificial_intelligence\third\ResNet\kaggle_dalao\PhanDai_dalao'

# ===== ResNet18 手写实现（完全兼容torchvision预训练权重） =====
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=120):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def build_resnet18(num_classes=120):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

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

# ====== 模型初始化及加载预训练权重 =======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_resnet18(num_classes=1000)  # 先用1000类（兼容官方权重）
import torchvision
pretrained = torchvision.models.resnet18(pretrained=True)
model.load_state_dict(pretrained.state_dict(), strict=False)
# 替换最后fc层为本任务类别数
model.fc = nn.Linear(512, len(encoder.classes_))
model = model.to(device)

# ====== 损失函数和优化器 =======
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# ====== 训练主循环 =======
train_losses, val_losses = [], []
epochs = 30
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for imgs, labels_ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - train"):
        imgs = imgs.to(device)
        #labels_ = labels_.to(device)
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
            #labels_ = labels_.to(device)
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
    torch.save(model.state_dict(), os.path.join(save_dir, f'resnet18_epoch{epoch+1}.pth'))

plt.figure(figsize=(10,6))
plt.plot(range(1, epochs+1), train_losses, label='train loss')
plt.plot(range(1, epochs+1), val_losses, label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training_Validation Loss')
plt.grid(True)

plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
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
