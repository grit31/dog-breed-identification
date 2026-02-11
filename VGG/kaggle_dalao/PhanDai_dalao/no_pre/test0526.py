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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score

# ===== 路径配置 =====
data_dir = r'D:\homework\Course_Design_of_artificial_intelligence\third\dog-breed-identification'
save_dir = r'D:\homework\Course_Design_of_artificial_intelligence\third\VGG\kaggle_dalao\PhanDai_dalao\no_pre\test0527'

# ===== 动态VGG结构配置 =====
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

class VGG(nn.Module):
    def __init__(self, features, num_classes=120):
        super().__init__()
        self.features = features
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def build_vgg(cfg, num_classes=120, batch_norm=False):
    features = make_layers(cfg, batch_norm=batch_norm)
    return VGG(features, num_classes=num_classes)

# ====== 加载标签 =======
labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
labels['image_path'] = labels['id'].apply(lambda x: os.path.join(data_dir, 'train', f'{x}.jpg'))
encoder = LabelEncoder()
labels['breed'] = encoder.fit_transform(labels['breed'])

# ====== 数据划分 =======
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

# ====== 模型初始化 =======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_vgg(cfg_vgg11, num_classes=len(encoder.classes_), batch_norm=False)
model = model.to(device)

# ====== 损失函数和优化器 =======
criterion = nn.CrossEntropyLoss()
#optimizer = optim.AdamW(model.parameters(), lr=0.0001)
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.01)
#scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# ====== 训练主循环（加入评价与历史记录）=======
train_losses, val_losses = [], []
val_accs, val_top5s, val_f1s = [], [], []
epochs = 80
for epoch in range(epochs):
    model.train()
    train_loss = 0
    all_preds, all_targets = [], []
    for imgs, labels_ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - train"):
        imgs = imgs.to(device)
        labels_ = labels_.to(device).long()
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels_)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        # 累加所有训练预测和标签
        preds = output.argmax(1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels_.cpu().numpy())
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # 验证
    model.eval()
    val_loss = 0
    preds_list, targets_list, probs_list = [], [], []
    with torch.no_grad():
        for imgs, labels_ in tqdm(val_loader, desc="val"):
            imgs = imgs.to(device)
            labels_ = labels_.to(device).long()
            output = model(imgs)
            loss = criterion(output, labels_)
            val_loss += loss.item() * imgs.size(0)
            preds = output.argmax(1).detach().cpu().numpy()
            probs = torch.softmax(output, dim=1).cpu().numpy()
            preds_list.extend(preds)
            targets_list.extend(labels_.cpu().numpy())
            probs_list.append(probs)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    # ====== 评价指标 ======
    val_acc = accuracy_score(targets_list, preds_list)
    val_top5 = top_k_accuracy_score(targets_list, np.vstack(probs_list), k=5)
    val_f1 = f1_score(targets_list, preds_list, average='macro')
    val_accs.append(val_acc)
    val_top5s.append(val_top5)
    val_f1s.append(val_f1)
    scheduler.step()
    print(f'Epoch {epoch+1}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, val acc {val_acc:.4f}, val top5 {val_top5:.4f}, val f1 {val_f1:.4f}')
    torch.save(model.state_dict(), os.path.join(save_dir, f'vgg11_epoch{epoch+1}.pth'))

# ====== 可视化训练过程（增）======
plt.figure(figsize=(10,6))
plt.plot(range(1, epochs+1), train_losses, label='train_loss')
plt.plot(range(1, epochs+1), val_losses, label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training/Validation Loss')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'Training_Validation_Loss.png'))
plt.close()

plt.figure(figsize=(10,6))
plt.plot(range(1, epochs+1), val_accs, label='val_acc')
plt.plot(range(1, epochs+1), val_top5s, label='val_top5')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.title('Validation Acc & Top-5')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'Validation_Acc_Top5.png'))
plt.close()

# ====== 保存历史指标到CSV（增） ======
pd.DataFrame({
    'epoch': np.arange(1, epochs+1),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'val_acc': val_accs,
    'val_top5': val_top5s,
    'val_f1': val_f1s
}).to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)

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
