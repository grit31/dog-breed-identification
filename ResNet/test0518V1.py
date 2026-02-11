import argparse
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ========== 1. 残差块 & ResNet18 主体 ==========
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
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
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
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

def resnet18(num_classes=120):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# ========== 2. Dataset ==========
class DogBreedDataset(Dataset):
    def __init__(self, df, img_dir, label_encoder=None, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, img_id
        else:
            label = self.df.iloc[idx, 1]
            label = self.label_encoder.transform([label])[0]
            return image, label

# ========== 3. Multi-class log loss ==========
def multiclass_logloss(y_true, y_pred, eps=1e-15):
    """y_true: shape (n_samples,), y_pred: shape (n_samples, n_classes)"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    y_pred /= y_pred.sum(axis=1, keepdims=True)
    N = y_pred.shape[0]
    ll = -np.log(y_pred[range(N), y_true]).mean()
    return ll

# ========== 4. Main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='cpu or cuda')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_dir', type=str, default='./train')
    parser.add_argument('--test_dir', type=str, default='./test')
    parser.add_argument('--labels_csv', type=str, default='./labels.csv')
    parser.add_argument('--sample_submission_csv', type=str, default='./sample_submission.csv')
    parser.add_argument('--output_csv', type=str, default='./submission.csv')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # 1. 数据读取
    labels_df = pd.read_csv(args.labels_csv)
    test_df = pd.read_csv(args.sample_submission_csv)[['id']]

    label_encoder = LabelEncoder()
    label_encoder.fit(labels_df['breed'])
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")

    # 2. 图像增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 3. Dataset & DataLoader
    train_dataset = DogBreedDataset(labels_df, args.train_dir, label_encoder, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 4. Model
    model = resnet18(num_classes=num_classes).to(device)

    # 5. Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # 6. Train Loop
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()
        print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader.dataset):.4f}")

    # 7. Score (Multi-class log loss) on training set
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            y_pred.append(probs)
            y_true.extend(labels.numpy())
    y_pred = np.vstack(y_pred)
    train_score = multiclass_logloss(np.array(y_true), y_pred)
    print(f"Train Multi-Class Log Loss: {train_score:.4f}")

    # 8. Inference on test set & 时间测试
    test_dataset = DogBreedDataset(test_df, args.test_dir, transform=val_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    all_img_ids, all_probs = [], []
    total_time, total_imgs = 0.0, 0
    for images, img_ids in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        start = time.time()
        outputs = model(images)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        used = (time.time() - start) * 1000  # 毫秒
        total_time += used
        total_imgs += images.size(0)
        all_img_ids.extend(img_ids)
        all_probs.append(probs)
    all_probs = np.vstack(all_probs)
    avg_time = total_time / total_imgs
    print(f"Average test time: {avg_time:.2f} ms/image")

    # 9. 输出CSV
    sub_df = pd.read_csv(args.sample_submission_csv)
    sub_df.set_index('id', inplace=True)
    for i, class_name in enumerate(label_encoder.classes_):
        sub_df[class_name] = all_probs[:, i]
    sub_df = sub_df.loc[all_img_ids]  # 保证顺序
    sub_df.reset_index(inplace=True)
    sub_df.to_csv(args.output_csv, index=False)
    print(f"Submission saved to {args.output_csv}")

if __name__ == "__main__":
    main()
