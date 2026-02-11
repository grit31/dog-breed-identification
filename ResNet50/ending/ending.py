import os
import time
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from thop import profile  # pip install thop
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score
import psutil

# ========== 1. Dataset 类 ==========
class DogBreedDataset(Dataset):
    """用于训练/验证的数据集定义"""
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
    """用于Kaggle提交/测试集推理的数据集定义"""
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

# ========== 2. 统计/画图/推理辅助 ==========
def count_parameters(model):
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_mb(model):
    """统计模型大小（MB）"""
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb

def get_flops(model, input_size=(1, 3, 224, 224)):
    """用thop统计模型FLOPs"""
    input = torch.randn(input_size).to(next(model.parameters()).device)
    macs, params = profile(model, inputs=(input,), verbose=False)
    flops = 2 * macs
    return flops, params

def get_memory_usage(device):
    """获取最大显存占用或内存占用（MB）"""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device=device) / 1024 ** 2
    else:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

def train_epoch(model, dataloader, criterion, optimizer, device):
    """单个epoch的训练"""
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
    # Top-5准确率
    top5 = top_k_accuracy_score(targets_all, np.vstack([np.eye(outputs.shape[1])[pred] for pred in preds_all]), k=5)
    return loss, acc, top5

def validate(model, dataloader, criterion, device):
    """单个epoch的验证"""
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
    """推理平均时间统计（ms/图像）"""
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

def plot_metrics(history, save_dir, suffix="resnet50"):
    """训练loss/acc曲线保存"""
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

# ========== 3. 主程序 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fff', help='dummy arg for notebook', default=None)
    parser.add_argument('--data_dir', type=str, default=r'D:\homework\Course_Design_of_artificial_intelligence\third\dog-breed-identification')
    parser.add_argument('--save_dir', type=str, default=r'D:\homework\Course_Design_of_artificial_intelligence\third\ResNet50\ending')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--augment', type=bool, default=True)
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=True)
    parser.add_argument('--model_path', type=str, default='resnet50_best.pth')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    args.device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Using device: {args.device}")
    print(f"数据增强：{args.augment} | 训练：{args.do_train} | 测试：{args.do_test}")

    # ====== 1. 读取标签和数据集分割 =======
    labels = pd.read_csv(os.path.join(args.data_dir, 'labels.csv'))
    labels['image_path'] = labels['id'].apply(lambda x: os.path.join(args.data_dir, 'train', f'{x}.jpg'))
    encoder = LabelEncoder()
    labels['breed'] = encoder.fit_transform(labels['breed'])
    train_df, val_df = train_test_split(
        labels, test_size=0.2, random_state=42, stratify=labels['breed'])

    # ====== 2. 数据增强与加载 =======
    if args.augment:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
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
    num_classes = len(encoder.classes_)
    # 加载ResNet50预训练模型并替换最后一层
    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(args.device)
    # 冻结除最后全连接层以外的所有层
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError('不支持的优化器类型')
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # ====== 4. 统计参数/FLOPs/模型大小 =======
    print("------ 模型结构和参数量 ------")
    summary(model, input_size=(3, 224, 224), device=str(args.device))
    total_params = count_parameters(model)
    print(f"参数量: {total_params / 1e6:.2f}M")
    model_mb = model_size_mb(model)
    print(f"模型大小: {model_mb:.2f} MB")
    import copy
    model_for_profile = copy.deepcopy(model)
    flops, _ = get_flops(model_for_profile, input_size=(1, 3, 224, 224))
    del model_for_profile
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs/图像")

    # ====== 5. 训练 =======
    if args.do_train:
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_top5': [], 'val_f1': []}
        best_loss = float('inf')
        best_epoch = 0
        start_train = time.time()
        for epoch in range(args.epochs):
            print(f"\n==== Epoch {epoch+1}/{args.epochs} ====")
            train_loss, train_acc, train_top5 = train_epoch(model, train_loader, criterion, optimizer, args.device)
            val_loss, val_acc, val_top5, val_f1 = validate(model, val_loader, criterion, args.device)
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Top-5: {val_top5:.4f} | F1: {val_f1:.4f}")
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_top5'].append(val_top5)
            history['val_f1'].append(val_f1)
            # 保存val loss最小的模型
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.model_path))
            scheduler.step()
        total_train_time = time.time() - start_train
        print(f"收敛到最优Val Loss Epoch: {best_epoch}，总训练用时: {total_train_time/60:.2f} 分钟")
        plot_metrics(history, args.save_dir, suffix="resnet50")
        pd.DataFrame(history).to_csv(os.path.join(args.save_dir, 'train_history_resnet50.csv'), index=False)

    # ====== 6. 测试与测试耗时/显存统计 =======
    if args.do_test:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.model_path), map_location=args.device))
        model.eval()
        # 统计推理速度/显存
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
        submission.to_csv(os.path.join(args.save_dir, 'submission_resnet50.csv'), index=False)
        print(f"Kaggle提交文件保存至: {os.path.join(args.save_dir, 'submission_resnet50.csv')}")

if __name__ == "__main__":
    main()
