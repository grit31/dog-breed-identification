import os
import time
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from thop import profile  # pip install thop
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score

# ========== 1. SE Block 定义 ==========
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

# ========== 2. BasicBlock + SE ==========
class BasicBlockSE(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, reduction=16):
        super(BasicBlockSE, self).__init__()
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

# ========== 3. ResNet18-SE整体 ==========
class ResNetSE(nn.Module):
    def __init__(self, block, layers, num_classes=120, reduction=16):
        super(ResNetSE, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, reduction=reduction)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction=reduction)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride, reduction):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample, reduction)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, reduction=reduction))
        return nn.Sequential(*layers)

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)

    def _initialize_weights(self):
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

def build_resnet18_se(num_classes=120, reduction=16):
    return ResNetSE(BasicBlockSE, [2, 2, 2, 2], num_classes=num_classes, reduction=reduction)

# ========== 4. 数据集定义 ==========
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

class DogBreedTestDataset(Dataset):
    def __init__(self, test_dir, test_ids, transform=None):
        self.test_dir = test_dir
        self.test_ids = test_ids
        self.transform = transform
    def __len__(self):
        return len(self.test_ids)
    def __getitem__(self, idx):
        img_id = self.test_ids[idx]
        img_path = os.path.join(self.test_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_id

# ========== 5. 训练&验证&测试&可视化，与VGG11/ResNet对齐 ==========
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

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    preds_all, targets_all = [], []
    for imgs, labels in tqdm(dataloader, desc="Train"):
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(outputs.detach(), dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        preds_all.extend(preds)
        targets_all.extend(labels.cpu().numpy())
    loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(targets_all, preds_all)
    top5 = top_k_accuracy_score(targets_all, np.eye(len(set(targets_all)))[preds_all], k=5)
    return loss, acc, top5

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    preds_all, targets_all, probs_all = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Val"):
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            preds_all.extend(preds)
            targets_all.extend(labels.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
    loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(targets_all, preds_all)
    top5 = top_k_accuracy_score(targets_all, np.vstack(probs_all), k=5)
    f1 = f1_score(targets_all, preds_all, average="macro")
    return loss, acc, top5, f1

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

# ========== 6. 主程序入口 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fff', help='a dummy argument for Jupyter notebook', default=None)
    parser.add_argument('--data_dir', type=str, default=r'D:\homework\Course_Design_of_artificial_intelligence\third\dog-breed-identification')
    parser.add_argument('--save_dir', type=str, default=r'D:\homework\Course_Design_of_artificial_intelligence\third\ResNet_Seblock\fangzhang_ResNet_kaggle_dalao\sam_dalao_before_seblock\newV1_sgd')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'], help='选择优化器（adam 或 sgd）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--augment', action='store_true', help='是否启用数据增强（不加参数则为无增强）')
    parser.add_argument('--do_train', type=bool, default=True, help='是否训练，默认True')
    parser.add_argument('--do_test', type=bool, default=True, help='是否测试，默认True')
    parser.add_argument('--model_path', type=str, default='resnet18se_sgd_30.pth')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--reduction', type=int, default=16)
    args = parser.parse_args()

    args.device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Using device: {args.device}")
    print(f"数据增强：{'启用' if args.augment else '关闭'} | 训练：{args.do_train} | 测试：{args.do_test}")

    if args.optimizer == 'adam' and args.lr == 1e-3:
        args.lr = 0.001
    elif args.optimizer == 'sgd' and args.lr == 1e-2:
        args.lr = 0.01

    # ====== 1. 读取标签和数据集分割 =======
    labels = pd.read_csv(os.path.join(args.data_dir, 'labels.csv'))
    sample_submission = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    labels['image_path'] = labels['id'].apply(lambda x: os.path.join(args.data_dir, 'train', f'{x}.jpg'))
    encoder = LabelEncoder()
    labels['breed'] = encoder.fit_transform(labels['breed'])
    train_df, val_df = train_test_split(
        labels, test_size=0.2, random_state=42, stratify=labels['breed'])

    # ====== 2. 数据增强与加载 =======
    if args.augment:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = DogBreedDataset(train_df, transform=train_transform)
    val_dataset = DogBreedDataset(val_df, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ====== 3. 模型、优化器、损失、调度器 =======
    model = build_resnet18_se(num_classes=len(encoder.classes_), reduction=args.reduction).to(args.device)
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError('不支持的优化器类型')
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # ====== 4. 统计参数/FLOPs/模型大小 =======
    print("------ 模型结构和参数量 ------")
    summary(model, input_size=(3, 224, 224), device=str(args.device))
    total_params = count_parameters(model)
    print(f"参数量: {total_params / 1e6:.2f}M")
    model_mb = model_size_mb(model)
    print(f"模型大小: {model_mb:.2f} MB")
    flops, _ = get_flops(model, input_size=(1, 3, 224, 224))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs/图像")

    # ====== 5. 训练 =======
    if args.do_train:
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_top5': [], 'val_f1': []}
        best_loss = float('inf')
        best_epoch = 0
        start_train = time.time()
        for epoch in range(args.epochs):
            print(f"\n==== Epoch {epoch+1}/{args.epochs} ====")
            train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer, args.device)
            val_loss, val_acc, val_top5, val_f1 = validate(model, val_loader, criterion, args.device)
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Top-5: {val_top5:.4f} | F1: {val_f1:.4f}")
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_top5'].append(val_top5)
            history['val_f1'].append(val_f1)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.model_path))
            scheduler.step()
        total_train_time = time.time() - start_train
        print(f"收敛到最优Val Loss Epoch: {best_epoch}，总训练用时: {total_train_time/60:.2f} 分钟")
        plot_metrics(history, args.save_dir, suffix="se")
        pd.DataFrame(history).to_csv(os.path.join(args.save_dir, 'train_history_se.csv'), index=False)

    # ====== 6. 测试与测试耗时/显存统计 =======
    if args.do_test:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.model_path), map_location=args.device))
        model.eval()
        if args.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=args.device)
        val_loader_speed = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        avg_time = test_speed(model, val_loader_speed, args.device)
        mem_usage = get_memory_usage(args.device)
        print(f"推理平均测试速度: {avg_time:.3f} ms/图像")
        print(f"推理最大显存占用: {mem_usage:.2f} MB")

        # Kaggle提交文件生成
        test_dir = os.path.join(args.data_dir, 'test')
        test_ids = [f.split('.')[0] for f in os.listdir(test_dir) if f.endswith('.jpg')]
        test_dataset = DogBreedTestDataset(test_dir, test_ids, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        all_predictions = []
        with torch.no_grad():
            for imgs, img_names in tqdm(test_loader, desc="Kaggle Test"):
                imgs = imgs.to(args.device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                for i, img_name in enumerate(img_names):
                    all_predictions.append([img_name] + probs[i].tolist())
        sub_columns = ['id'] + list(encoder.classes_)
        submission = pd.DataFrame(all_predictions, columns=sub_columns)
        submission.to_csv(os.path.join(args.save_dir, 'submission_se.csv'), index=False)
        print(f"Kaggle提交文件保存至: {os.path.join(args.save_dir, 'submission_se.csv')}")

if __name__ == "__main__":
    main()
