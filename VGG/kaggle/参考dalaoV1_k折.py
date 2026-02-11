import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from skimage import io

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import KFold

# --- 路径 ---
save_dir = './k'

# --- VGG11 构建 ---
cfg_vgg11 = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
class VGG(nn.Module):
    def __init__(self, features, num_classes=120):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    @staticmethod
    def make_layers(cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ]
                in_channels = v
        return nn.Sequential(*layers)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
def get_vgg11(num_classes):
    return VGG(VGG.make_layers(cfg_vgg11), num_classes=num_classes)

# --- Dataset ---
class DogBreedsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.map = dict(zip(self.labels_frame['breed'].unique(), range(len(self.labels_frame['breed'].unique()))))
        self.inv_map = {v: k for k, v in self.map.items()}
        self.labels_frame['breed'] = self.labels_frame['breed'].map(self.map)
        self.root_dir = root_dir
        self.transform = transform
    def getmap(self):
        return self.map
    def get_invmap(self):
        return self.inv_map
    def __getclasses__(self):
        return sorted(self.labels_frame['breed'].unique().tolist())
    def __len__(self):
        return len(self.labels_frame)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0] + '.jpg')
        image = io.imread(img_name)
        PIL_image = Image.fromarray(image)
        label = self.labels_frame.iloc[idx, 1]
        label = int(label)
        if self.transform:
            image = self.transform(PIL_image)
        return image, label

class DogBreedsTestset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.labels_frame = self.labels_frame[['id']]
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.labels_frame)
    def __getitem__(self, idx):
        title = self.labels_frame.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, title + '.jpg')
        image = io.imread(img_name)
        PIL_image = Image.fromarray(image)
        if self.transform:
            image = self.transform(PIL_image)
        return {'image': image, 'title': title}

def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

# --- 参数设置 ---
data_dir = '../../dog-breed-identification'
batch_size = 32
valid_size = 0.2
num_epochs = 50
lr = 0.0005
num_folds = 2
train_on_gpu = torch.cuda.is_available()
device = torch.device('cuda' if train_on_gpu else 'cpu')
print(f'Using device: {device}')

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_data = DogBreedsDataset(csv_file=os.path.join(data_dir,'labels.csv'), root_dir=os.path.join(data_dir,'train'), transform=transform)
classes = train_data.__getclasses__()
inv_map = train_data.get_invmap()
print('类别数量:', len(classes))
print('类别索引->类别名:', {i:inv_map[i] for i in range(len(inv_map))})

# --- 测试集 ---
test_data = DogBreedsTestset(csv_file=os.path.join(data_dir,'sample_submission.csv'), root_dir=os.path.join(data_dir,'test'), transform=transform)
test_loader = DataLoader(test_data, batch_size=batch_size)

# --- K折训练与验证 ---
num_train = len(train_data)
indices = np.arange(num_train)
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_preds = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(indices)):
    print(f"\n========== Fold {fold+1}/{num_folds} ==========")
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)

    model = get_vgg11(num_classes=len(classes)).to(device)
    for param in model.features.parameters():
        param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    train_loss_history, valid_loss_history = [], []
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs+1):
        print(f"==== Fold {fold+1} Epoch {epoch}/{num_epochs} (Device: {device}) ====")
        model.train()
        train_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Fold{fold+1}', ncols=100)
        for batch_i, (data, target) in loop:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_i % 20 == 19:
                loop.set_postfix({'loss': train_loss/20})
                train_loss = 0.0

        # 验证
        model.eval()
        val_loss = 0.0
        val_loop = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'Valid Fold{fold+1}', ncols=100)
        with torch.no_grad():
            for batch_i, (data, target) in val_loop:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(valid_loader)
        valid_loss_history.append(avg_val_loss)
        print(f'[Fold {fold+1}] Epoch {epoch}, Validation Loss: {avg_val_loss:.6f}')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f'vgg11_best_fold{fold+1}.pth'))
            print(f'Best model saved at epoch {epoch} with val loss {avg_val_loss:.6f}')
        # 实时画loss曲线
        train_loss_history.append(train_loss)
        plt.figure()
        plt.plot(valid_loss_history, label='Val Loss')
        plt.title(f'Fold {fold+1} Validation Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'loss_curve_fold{fold+1}_epoch{epoch}.png'))
        plt.close()

    # 推理本折的test概率
    model.load_state_dict(torch.load(os.path.join(save_dir, f'vgg11_best_fold{fold+1}.pth'), map_location=device))
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Test Fold{fold+1}', ncols=100):
            images = batch['image'].to(device)
            logits = model(images)
            output = torch.nn.functional.softmax(logits, dim=1)
            preds.append(output.cpu().numpy())
    fold_preds.append(np.vstack(preds))

# --- 合并K折结果：所有折softmax概率平均 ---
final_preds = np.mean(fold_preds, axis=0)
test_ids = pd.read_csv(os.path.join(data_dir,'sample_submission.csv'))['id'].tolist()
output_df = pd.DataFrame(final_preds, columns=[inv_map[i] for i in range(len(classes))])
output_df.insert(0, 'id', test_ids)
output_df.to_csv(os.path.join(save_dir, 'submission_kfold.csv'), index=False)
print('KFold Test predictions saved to', os.path.join(save_dir, 'submission_kfold.csv'))
