import os
import time
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

# === 动态VGG11搭建 ===
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

# === 自定义Dataset ===
class DogDataset(Dataset):
    def __init__(self, df, img_dir, breed2idx, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.breed2idx = breed2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['id'] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.breed2idx[row['breed']]
        return img, label

class TestDataset(Dataset):
    def __init__(self, img_dir, img_ids, transform=None):
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_id

# === Log Loss计算 ===
def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1-eps)
    N = y_true.shape[0]
    ll = -np.sum(y_true * np.log(y_pred)) / N
    return ll

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='train')
    parser.add_argument('--test_dir', type=str, default='test')
    parser.add_argument('--labels', type=str, default='labels.csv')
    parser.add_argument('--sample_submission', type=str, default='sample_submission.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_csv', type=str, default='submission.csv')
    parser.add_argument('--do_train', action='store_true', help='train model')
    parser.add_argument('--do_test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, default='vgg11_dog.pth')
    args = parser.parse_args()

    print(f'Using device: {args.device}')

    # 1. 读取标签
    label_df = pd.read_csv(args.labels)
    breeds = sorted(label_df['breed'].unique())
    breed2idx = {breed: i for i, breed in enumerate(breeds)}
    idx2breed = {i: breed for breed, i in breed2idx.items()}
    num_classes = len(breeds)
    print(f'Classes: {num_classes}')

    # 2. 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # 3. 数据集
    train_dataset = DogDataset(label_df, args.train_dir, breed2idx, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 4. 构建模型
    model = get_vgg11(num_classes)
    model = model.to(args.device)

    if args.do_train:
        # 5. 训练
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        model.train()
        for epoch in range(args.epochs):
            running_loss = 0.0
            for images, labels in tqdm(train_loader):
                images = images.to(args.device)
                labels = labels.to(args.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), args.model_path)
        print("Model saved.")

    if args.do_test:
        # 6. 推理与生成提交
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        model.eval()
        # 获取测试集ID
        submission_df = pd.read_csv(args.sample_submission)
        test_ids = submission_df['id'].tolist()
        test_dataset = TestDataset(args.test_dir, test_ids, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        preds = []
        ids = []
        total_imgs = 0
        total_time = 0
        with torch.no_grad():
            for images, img_ids in tqdm(test_loader):
                images = images.to(args.device)
                batch_size = images.size(0)
                start = time.time()
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                infer_time = (time.time() - start) * 1000 # ms
                total_time += infer_time
                total_imgs += batch_size
                preds.append(probs.cpu().numpy())
                ids.extend(img_ids)
        preds = np.vstack(preds)
        print(f"Avg test speed: {total_time/total_imgs:.3f} ms/image")

        # 写入CSV
        df_out = pd.DataFrame(preds, columns=breeds)
        df_out.insert(0, "id", ids)
        df_out.to_csv(args.output_csv, index=False)
        print(f"Saved result to {args.output_csv}")

        # 如果你有labels.csv和test集的真实标签，可计算Score
        # （这里只是模板，如需实际score，请提供test真实标签one-hot和预测概率）

if __name__ == "__main__":
    main()
