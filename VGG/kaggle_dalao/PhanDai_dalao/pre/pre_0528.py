import os
import time
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import psutil
from thop import profile
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# ===== 路径配置 =====
data_dir = r'D:\homework\Course_Design_of_artificial_intelligence\third\dog-breed-identification'
save_dir = r'D:\homework\Course_Design_of_artificial_intelligence\third\VGG\kaggle_dalao\PhanDai_dalao\pre'
os.makedirs(save_dir, exist_ok=True)

# ===== VGG11结构 =====
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

# ====== 模型初始化及加载预训练权重 =======
import torchvision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG11(num_classes=1000, batch_norm=False)
state_dict = torchvision.models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1').state_dict()
model.load_state_dict(state_dict)
model.classifier[-1] = nn.Linear(4096, 120)
model = model.to(device)

# ========== 统计参数/模型大小/FLOPs ==========
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def model_size_mb(model):
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb
def get_flops(model, input_size=(1, 3, 224, 224)):
    x = torch.randn(input_size).to(device)
    macs, params = profile(model, inputs=(x,), verbose=False)
    flops = 2 * macs
    return flops, params

print("\n=== Model Info ===")
print(f"参数量: {count_parameters(model)/1e6:.2f} M")
print(f"模型大小: {model_size_mb(model):.2f} MB")
flops, _ = get_flops(model, input_size=(1, 3, 224, 224))
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs/图像\n")

# ====== 损失函数和优化器 =======
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# ====== 训练主循环（保存val_loss最小的权重） =======
epochs = 30
train_losses, val_losses = [], []
val_accs, val_top5s = [], []
best_loss = float('inf')
best_epoch = -1
start_train = time.time()

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
    top5_correct = 0
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
            top5_correct += (output.topk(5, 1)[1] == labels_.view(-1, 1)).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc = correct / total
    val_top5 = top5_correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_top5s.append(val_top5)
    scheduler.step()
    print(f'Epoch {epoch+1}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, val acc {val_acc:.4f}, val top5 {val_top5:.4f}')

    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(save_dir, 'vgg11_best.pth'))
        print(f">>> Model saved at epoch {best_epoch}, val_loss = {best_loss:.5f}")

total_time = time.time() - start_train
print(f'\n训练总耗时: {total_time/60:.2f} 分钟，平均每epoch耗时: {total_time/epochs:.2f} 秒')
print(f"Best epoch: {best_epoch}, best val_loss: {best_loss:.5f}")

# ========== 损失/精度曲线可视化 ==========
def plot_metrics(history, save_dir, suffix="vgg11"):
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

history = {'train_loss': train_losses, 'val_loss': val_losses, 'val_acc': val_accs, 'val_top5': val_top5s}
plot_metrics(history, save_dir, suffix="vgg11")

# ========== 验证集评估 ==========
model.load_state_dict(torch.load(os.path.join(save_dir, 'vgg11_best.pth')))
model.eval()
y_true, y_pred, y_pred_logits = [], [], []
for imgs, labels_ in tqdm(val_loader, desc="Final Valid Eval"):
    imgs = imgs.to(device)
    labels_ = labels_.to(device)
    with torch.no_grad():
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        y_true.extend(labels_.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_pred_logits.extend(logits.cpu().numpy())
val_top1_acc = accuracy_score(y_true, y_pred)
val_top5_acc = top_k_accuracy_score(y_true, np.vstack(y_pred_logits), k=5)
print(f"\nFinal Validation Top-1 Acc: {val_top1_acc:.4f}, Top-5 Acc: {val_top5_acc:.4f}")

# ========== 推理速度与内存占用 ==========
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
    return (total_time / n) * 1000

def get_memory_usage(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device=device) / 1024 ** 2
    else:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device=device)
speed = test_speed(model, val_loader, device)
mem_usage = get_memory_usage(device)
print(f"\n推理平均测试速度: {speed:.3f} ms/图像")
print(f"推理最大显存占用: {mem_usage:.2f} MB")

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
