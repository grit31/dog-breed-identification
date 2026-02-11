import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import torch.optim as optim

# ========== 自动路径 ==========
data_dir = '../input'
save_dir = './'

# ========== CUDA 检查 ==========
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

# ========== VGG11 配置 ==========
cfg_vgg11 = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

def make_layers(cfg, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, features, num_classes=120):  # num_classes需根据数据集动态调整
        super(VGG, self).__init__()
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

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ========== 数据集定义 ==========
class DogBreedsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.map = dict(zip(self.labels_frame['breed'].unique(), range(len(self.labels_frame['breed'].unique()))))
        self.labels_frame['breed'] = self.labels_frame['breed'].map(self.map)
        self.root_dir = root_dir
        self.transform = transform

    def getmap(self):
        return self.map

    def __getclasses__(self):
        return self.labels_frame['breed'].unique().tolist()

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0] + '.jpg')
        image = io.imread(img_name)
        PIL_image = Image.fromarray(image)
        label = int(self.labels_frame.iloc[idx, 1])
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

# ========== 数据变换 ==========
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ========== 数据加载 ==========
train_data = DogBreedsDataset(csv_file=os.path.join(data_dir, 'labels.csv'), root_dir=os.path.join(data_dir, 'train'), transform=transform)
classes = train_data.__getclasses__()
num_classes = len(classes)
print("num_classes:", num_classes)

valid_size = 0.2
batch_size = 20
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)

df_test = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
test_data = DogBreedsTestset(csv_file=os.path.join(data_dir, 'sample_submission.csv'), root_dir=os.path.join(data_dir, 'test'), transform=transform)
test_loader = DataLoader(test_data, batch_size=20)

# ========== VGG11 实例化 ==========
vgg11 = VGG(make_layers(cfg_vgg11), num_classes=num_classes)
if train_on_gpu:
    vgg11.cuda()
print(vgg11)

# ========== 损失函数与优化器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg11.parameters(), lr=0.001, momentum=0.9)

# ========== 训练 ==========
n_epochs = 15
for epoch in range(1, n_epochs + 1):
    vgg11.train()
    train_loss = 0.0
    for batch_i, (data, target) in enumerate(train_loader):
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = vgg11(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_i % 20 == 19:
            print('Epoch %d, Batch %d loss: %.6f' % (epoch, batch_i + 1, train_loss / 20))
            train_loss = 0.0

# ========== 验证 ==========
vgg11.eval()
valid_loss = 0.0
for batch_i, (data, target) in enumerate(valid_loader):
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    output = vgg11(data)
    loss = criterion(output, target)
    valid_loss += loss.item()
    if batch_i % 20 == 19:
        print('Validation Loss Batch %d loss: %.6f' % (batch_i + 1, valid_loss / 20))
        valid_loss = 0.0

# ========== 测试预测 ==========
results = {}
vgg11.eval()
inv_map = {v: k for k, v in train_data.getmap().items()}

with torch.no_grad():
    for data in test_loader:
        images, titles = data['image'], data['title']
        if train_on_gpu:
            images = images.cuda()
        logits = vgg11(images)
        output = torch.nn.functional.softmax(logits, dim=1)
        for k in range(len(titles)):
            name = titles[k]
            results[name] = output[k].cpu().tolist()

output_df = pd.DataFrame(results).transpose()
output_df.rename(columns=inv_map, inplace=True)
output_df = output_df.reset_index()
output_df.rename(columns={'index':'id'}, inplace=True)
output_df.to_csv(os.path.join(save_dir, 'output.csv'), index=False)
