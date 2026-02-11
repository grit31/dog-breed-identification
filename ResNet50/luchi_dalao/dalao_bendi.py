
import os
import time
import copy
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
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from thop import profile  # pip install thop
import psutil
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# ==================== 路径设置 ====================
base_dir = r'D:\homework\Course_Design_of_artificial_intelligence\third\dog-breed-identification'
save_dir = r'D:\homework\Course_Design_of_artificial_intelligence\third\ResNet50\luchi_dalao\dalao_bendi0529V1'
os.makedirs(save_dir, exist_ok=True)

labels_path = os.path.join(base_dir, 'labels.csv')
train_img_path = os.path.join(base_dir, 'train')
test_img_path = os.path.join(base_dir, 'test')

# ==================== 读取并分析标签 ====================
breeds_dataframe = pd.read_csv(labels_path)
dog_breeds = sorted(list(set(breeds_dataframe['breed'])))
n_breeds = len(dog_breeds)
breed_to_num = dict(zip(dog_breeds, range(n_breeds)))

# ==================== 自定义Dataset ====================
class DogDataset(Dataset):
    def __init__(self, csv_path, file_path, mode="train", transform=None, valid_ratio=0.2):
        self.file_path = file_path
        self.mode = mode
        self.transform = transform

        if csv_path is not None:
            self.data_info = pd.read_csv(csv_path)
            self.data_len = len(self.data_info.index)
            self.train_len = int(self.data_len * (1 - valid_ratio))
        if mode == "train":
            self.image_arr = np.asarray(self.data_info.iloc[0:self.train_len, 0])
            self.breed_arr = np.asarray(self.data_info.iloc[0:self.train_len, 1])
        elif mode == "valid":
            self.image_arr = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.breed_arr = np.asarray(self.data_info.iloc[self.train_len:, 1])
        elif mode == 'test':
            self.image_arr = [f.split('.')[0] for f in os.listdir(file_path) if f.endswith('.jpg')]
            self.breed_arr = None
        self.real_len = len(self.image_arr)
        print('Finished reading the {} set of Dog Dataset ({} samples found)'.format(mode, self.real_len))

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_path = os.path.join(self.file_path, single_image_name + ".jpg")
        img_as_img = Image.open(img_path)
        img_as_img = self.transform(img_as_img)
        if self.mode == "test":
            return img_as_img, single_image_name
        else:
            breed = self.breed_arr[index]
            number_breed = breed_to_num[breed]
            return img_as_img, number_breed

    def __len__(self):
        return self.real_len

# ========== 数据增强和预处理 ==========
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== 数据集与DataLoader ==========
train_dataset = DogDataset(labels_path, train_img_path, mode="train", transform=train_transform)
valid_dataset = DogDataset(labels_path, train_img_path, mode='valid', transform=test_transform)
test_dataset = DogDataset(None, test_img_path, mode='test', transform=test_transform)
batch_size = 128
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
valid_iter = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# ========== 设备和网络 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def get_net(device):
    net = torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT")
    # 替换最后一层
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, n_breeds)
    net = net.to(device)
    for name, param in net.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    return net

learning_rate = 1e-4
weight_decay = 1e-3
num_epoch = 100
model_path = os.path.join(save_dir, 'resnet50_best.pth')
model = get_net(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# ====== 统计模型参数、模型大小、FLOPs（只profile一次副本） ======
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def model_size_mb(model):
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb
def get_flops(model, input_size=(1, 3, 224, 224)):
    input = torch.randn(input_size).to(device)
    macs, params = profile(model, inputs=(input,), verbose=False)
    flops = 2 * macs
    return flops, params

import copy
model_for_profile = copy.deepcopy(model)
flops, _ = get_flops(model_for_profile, input_size=(1, 3, 224, 224))
del model_for_profile
print("\n=== Model Info ===")
print(f"参数量: {count_parameters(model)/1e6:.2f} M")
print(f"模型大小: {model_size_mb(model):.2f} MB")
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs/图像\n")

# ========== 训练主循环 ==========
total_train_loss, total_train_accs = [], []
total_valid_loss, total_valid_accs = [], []
train_top5s, valid_top5s = [], []
start_time = time.time()

best_loss = float('inf')
best_epoch = -1

for epoch in range(num_epoch):
    model.train()
    train_loss, train_accs, train_labels, train_preds = [], [], [], []
    for batch in tqdm(train_iter, desc=f"Epoch {epoch+1}/{num_epoch} [Train]"):
        imgs, breeds = batch
        imgs = imgs.to(device)
        breeds = breeds.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_func(logits, breeds)
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=-1) == breeds).float().mean()
        train_loss.append(loss)
        train_accs.append(acc)
        train_labels.extend(breeds.cpu().numpy())
        train_preds.extend(logits.detach().cpu().numpy())
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    train_top5 = top_k_accuracy_score(train_labels, np.vstack(train_preds), k=5)
    total_train_loss.append(train_loss.item())
    total_train_accs.append(train_acc.item())
    train_top5s.append(train_top5)
    print(f"[Train | {epoch + 1:03d}/{num_epoch:03d}] loss = {train_loss:.5f}, acc = {train_acc:.5f}, top5 = {train_top5:.5f}")

    model.eval()
    valid_loss, valid_accs, valid_labels, valid_preds = [], [], [], []
    for batch in tqdm(valid_iter, desc=f"Epoch {epoch+1}/{num_epoch} [Valid]"):
        imgs, breeds = batch
        imgs = imgs.to(device)
        breeds = breeds.to(device)
        with torch.no_grad():
            logits = model(imgs)
        loss = loss_func(logits, breeds)
        acc = (logits.argmax(dim=-1) == breeds).float().mean()
        valid_loss.append(loss)
        valid_accs.append(acc)
        valid_labels.extend(breeds.cpu().numpy())
        valid_preds.extend(logits.detach().cpu().numpy())
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    valid_top5 = top_k_accuracy_score(valid_labels, np.vstack(valid_preds), k=5)
    total_valid_loss.append(valid_loss.item())
    total_valid_accs.append(valid_acc.item())
    valid_top5s.append(valid_top5)
    print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}, top5 = {valid_top5:.5f}")

    # 只在val loss最小时保存模型
    if valid_loss < best_loss:
        best_loss = valid_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_path)
        print(f">>> Model saved at epoch {best_epoch}, val_loss = {best_loss:.5f}")

total_time = time.time() - start_time
print(f'\n训练总耗时: {total_time/60:.2f} 分钟，平均每epoch耗时: {total_time/num_epoch:.2f} 秒')
print(f"Best epoch: {best_epoch}, best val_loss: {best_loss:.5f}")

# ========== 损失/精度曲线保存 ==========
def plot_metrics(history, save_dir, suffix="resnet50"):
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

history = {
    'train_loss': total_train_loss,
    'val_loss': total_valid_loss,
    'val_acc': total_valid_accs,
    'val_top5': valid_top5s
}
plot_metrics(history, save_dir, suffix="resnet50")

# ========== 验证集评估：Top-1/Top-5准确率 ==========
model.load_state_dict(torch.load(model_path))
model.eval()
y_true, y_pred, y_pred_logits = [], [], []
for batch in tqdm(valid_iter, desc="Final Valid Eval"):
    imgs, breeds = batch
    imgs = imgs.to(device)
    breeds = breeds.to(device)
    with torch.no_grad():
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        y_true.extend(breeds.cpu().numpy())
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
speed = test_speed(model, valid_iter, device)
mem_usage = get_memory_usage(device)
print(f"\n推理平均测试速度: {speed:.3f} ms/图像")
print(f"推理最大显存占用: {mem_usage:.2f} MB")

# ========== 预测并生成提交文件 ==========
model = get_net(device)
model.load_state_dict(torch.load(model_path))
model.eval()
preds = []
ids = []
for batch in tqdm(test_iter, desc="Test Inference"):
    imgs, imgs_id = batch
    imgs = imgs.to(device)
    with torch.no_grad():
        output = torch.nn.functional.softmax(model(imgs), dim=1)
        preds.extend(output.cpu().numpy())
        ids.extend(imgs_id)

submit_path = os.path.join(save_dir, 'submission.csv')
with open(submit_path, 'w', encoding='utf-8') as f:
    f.write('id,' + ','.join(dog_breeds) + '\n')
    for i, output in zip(ids, preds):
        f.write(i + ',' + ','.join([str(n) for n in output]) + '\n')
print(f"Submission file saved to {submit_path}")
