import os
import time
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

from torchsummary import summary
from thop import profile  # pip install thop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score

# ===== 路径配置 =====
data_dir = r'D:\homework\Course_Design_of_artificial_intelligence\third\dog-breed-identification'
save_dir = r'D:\homework\Course_Design_of_artificial_intelligence\third\ResNet_Seblock\fangzhang_ResNet_kaggle_dalao\PhanDai_ResNet_seblock_kaggle_dalao\fangzhao_dalao_V2'

# ===== SE Block =====
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ===== BasicBlock + SE =====
class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, reduction)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ===== SE-ResNet18 主体 =====
class SEResNet(nn.Module):
    def __init__(self, block, layers, num_classes=120, reduction=16):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], reduction=reduction)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction=reduction)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, blocks, stride=1, reduction=16):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, reduction)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction=reduction))
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

def build_se_resnet18(num_classes=120, reduction=16):
    return SEResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, reduction=reduction)

# ===== 工具函数 =====
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_mb(model):
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb

def get_flops(model, input_size=(1, 3, 224, 224)):
    input = torch.randn(input_size).to(next(model.parameters()).device)
    macs, params = profile(model, inputs=(input,), verbose=False)
    flops = 2 * macs
    return flops, params

def get_memory_usage(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device=device) / 1024 ** 2
    else:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

def test_speed(model, dataloader, device):
    model.eval()
    total_time = 0
    n = 0
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="SpeedTest"):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = model(imgs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += (time.time() - start)
            n += batch_size
    return (total_time / n) * 1000  # ms/图像

def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred, probs_all = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Eval"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
    top1 = accuracy_score(y_true, y_pred)
    top5 = top_k_accuracy_score(y_true, np.vstack(probs_all), k=5)
    return top1, top5

def plot_metrics(history, save_dir, suffix="se"):
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(save_dir, f"loss_curve_{suffix}.png"))
    plt.close()

    plt.figure()
    plt.plot(history['val_acc'], label='val_acc')
    plt.plot(history['val_top5'], label='val_top5')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(os.path.join(save_dir, f"acc_curve_{suffix}.png"))
    plt.close()

# ===== 加载标签 =====
labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
labels['image_path'] = labels['id'].apply(lambda x: os.path.join(data_dir, 'train', f'{x}.jpg'))
encoder = LabelEncoder()
labels['breed'] = encoder.fit_transform(labels['breed'])
train_df, val_df = train_test_split(
    labels, test_size=0.2, random_state=42, stratify=labels['breed'])

# ===== 数据集类 =====
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

# ===== 初始化模型并加载预训练权重 =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torchvision

model = build_se_resnet18(num_classes=1000).to(device)
pretrained_resnet18 = torchvision.models.resnet18(weights='IMAGENET1K_V1')
model.load_state_dict(pretrained_resnet18.state_dict(), strict=False)
model.fc = nn.Linear(512, len(encoder.classes_)).to(device)

# ===== 损失函数和优化器 =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)#3e-4
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# ===== 模型统计信息 =====
print("\n--- Model Summary ---")
summary(model, input_size=(3, 224, 224), device=str(device))
print(f"参数量: {count_parameters(model) / 1e6:.2f} M")
print(f"模型大小: {model_size_mb(model):.2f} MB")
flops, _ = get_flops(model, input_size=(1, 3, 224, 224))
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs/图像\n")

# ===== 训练主循环 =====
train_losses, val_losses = [], []
history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_top5': [], 'val_f1': []}
epochs = 30
best_loss = float('inf')
best_epoch = 0
start_train = time.time()
for epoch in range(epochs):
    model.train()
    train_loss, train_preds, train_targets = 0, [], []
    for imgs, labels_ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - train"):
        imgs = imgs.to(device)
        labels_ = labels_.to(device).long()
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels_)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        preds = output.argmax(1)
        train_preds.extend(preds.cpu().numpy())
        train_targets.extend(labels_.cpu().numpy())
    train_loss /= len(train_loader.dataset)
    train_acc = accuracy_score(train_targets, train_preds)
    train_losses.append(train_loss)

    # 验证
    model.eval()
    val_loss = 0
    val_preds, val_targets, val_probs = [], [], []
    with torch.no_grad():
        for imgs, labels_ in tqdm(val_loader, desc="val"):
            imgs = imgs.to(device)
            labels_ = labels_.to(device).long()
            output = model(imgs)
            loss = criterion(output, labels_)
            val_loss += loss.item() * imgs.size(0)
            preds = output.argmax(1)
            probs = F.softmax(output, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels_.cpu().numpy())
            val_probs.append(probs.cpu().numpy())
    val_loss /= len(val_loader.dataset)
    val_acc = accuracy_score(val_targets, val_preds)
    val_top5 = top_k_accuracy_score(val_targets, np.vstack(val_probs), k=5)
    val_f1 = f1_score(val_targets, val_preds, average="macro")
    val_losses.append(val_loss)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_top5'].append(val_top5)
    history['val_f1'].append(val_f1)
    scheduler.step()
    print(f'Epoch {epoch+1}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, val acc {val_acc:.4f}, val top5 {val_top5:.4f}, val f1 {val_f1:.4f}')
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(save_dir, f'seresnet18_best.pth'))
    torch.save(model.state_dict(), os.path.join(save_dir, f'seresnet18_epoch{epoch+1}.pth'))

total_train_time = time.time() - start_train
print(f"\n收敛到最优Val Loss Epoch: {best_epoch}，总训练用时: {total_train_time/60:.2f} 分钟")
plot_metrics(history, save_dir, suffix="se")
pd.DataFrame(history).to_csv(os.path.join(save_dir, 'train_history_se.csv'), index=False)

# ===== 验证集评估 =====
print("\n======= Final Validation Set Evaluation =======")
top1, top5 = evaluate(model, val_loader, device)
print(f"Validation Top-1 Accuracy: {top1:.4f}")
print(f"Validation Top-5 Accuracy: {top5:.4f}")

# ===== 推理时间、内存占用 =====
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device=device)
val_loader_speed = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
avg_time = test_speed(model, val_loader_speed, device)
mem_usage = get_memory_usage(device)
print(f"推理平均测试速度: {avg_time:.3f} ms/图像")
print(f"推理最大显存占用: {mem_usage:.2f} MB")

# ===== 测试集推理和Kaggle提交文件生成 =====
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
