import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import io
from tqdm import tqdm

# ========== 配置路径 ==========
data_dir = "../../dog-breed-identification"  # 数据文件夹路径
save_dir = "./"                              # 保存输出文件夹路径

# ========== CUDA 检查 ==========
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

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
        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0]) + '.jpg'
        image = io.imread(img_name)
        PIL_image = Image.fromarray(image)
        label = self.labels_frame.iloc[idx, 1:]
        label = [int(label) for x in label]
        label = np.asarray(label)
        label = torch.from_numpy(label)
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
        img_name = os.path.join(self.root_dir, title) + '.jpg'
        image = io.imread(img_name)
        PIL_image = Image.fromarray(image)
        if self.transform:
            image = self.transform(PIL_image)
        sample = {'image': image, 'title': title}
        return sample

# ========== 数据加载与增强 ==========
batch_size = 20
valid_size = 0.2

transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_csv = os.path.join(data_dir, "labels.csv")
train_img_dir = os.path.join(data_dir, "train")
test_csv = os.path.join(data_dir, "sample_submission.csv")
test_img_dir = os.path.join(data_dir, "test")

train_data = DogBreedsDataset(csv_file=train_csv, root_dir=train_img_dir, transform=transform)
classes = train_data.__getclasses__()

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)

test_data = DogBreedsTestset(csv_file=test_csv, root_dir=test_img_dir, transform=transform)
test_loader = DataLoader(test_data, batch_size=20)

# ========== 手动实现 VGG11 ==========
class VGG11(nn.Module):
    def __init__(self, num_classes=120, config=None, init_weights=True):
        super(VGG11, self).__init__()
        # VGG11: 配置A，8个conv + 3个fc
        # config: 可选, 支持动态指定每层通道数
        # 默认: 论文 Table1 配置A
        if config is None:
            config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.features = self._make_layers(config)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
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

# ========== 模型实例化 ==========
num_classes = len(classes)
# config 可调整：如[64, 'M', 128, ...]
model = VGG11(num_classes=num_classes)
if train_on_gpu:
    model.cuda()

# ========== 损失、优化器 ==========
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

n_epochs = 30

# ========== 训练主循环 ==========
for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    model.train()
    train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]", ncols=100)
    for batch_i, (data, target) in train_bar:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, torch.max(target, 1)[1])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_i % 20 == 19:
            train_bar.set_postfix(loss=(train_loss / 20))
            train_loss = 0.0

    valid_loss = 0.0
    model.eval()
    valid_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Epoch {epoch} [Valid]", ncols=100)
    with torch.no_grad():
        for batch_i, (data, target) in valid_bar:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, torch.max(target, 1)[1])
            valid_loss += loss.item()
            if batch_i % 20 == 19:
                valid_bar.set_postfix(loss=(valid_loss / 20))
                valid_loss = 0.0

# ========== 推理与保存 ==========
results = {}
model.eval()
with torch.no_grad():
    for data in tqdm(test_loader, desc="Predicting", ncols=100):
        images, titles = data['image'], data['title']
        if train_on_gpu:
            images = images.cuda()
        logits = model(images)
        output = torch.nn.functional.softmax(logits, dim=1)
        for k in range(len(titles)):
            name = titles[k]
            results[name] = output[k].cpu().tolist()

output_df = pd.DataFrame(results).transpose()
inv_map = {v: k for k, v in train_data.getmap().items()}
output_df.rename(columns=inv_map, inplace=True)
output_df = output_df.reset_index()
output_df.rename(columns={'index': 'id'}, inplace=True)

output_csv_path = os.path.join(save_dir, 'output.csv')
output_df.to_csv(output_csv_path, index=False)
print(f'Saved to {output_csv_path}')
