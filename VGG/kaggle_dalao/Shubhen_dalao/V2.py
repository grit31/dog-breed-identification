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

# 路径变量
data_dir = '../input'
save_dir = './'

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

# ================== 1. VGG11 手动实现 ==================
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
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

cfg_vgg11 = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self, features, num_classes=120, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
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
                nn.init.constant_(m.bias, 0)

def vgg11(num_classes=120, pretrained=False):
    model = VGG(make_layers(cfg_vgg11), num_classes=num_classes, init_weights=not pretrained)
    if pretrained:
        from torchvision import models
        state_dict = models.vgg11(pretrained=True).state_dict()
        model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict() and 'classifier.6' not in k}, strict=False)
    return model

# ================== 2. 数据集定义 ==================
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
        # 正确的标签格式：int
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

# ================== 3. 数据加载和增强 ==================
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = DogBreedsDataset(
    csv_file=os.path.join(data_dir, 'labels.csv'),
    root_dir=os.path.join(data_dir, 'train'),
    transform=transform
)
classes = train_data.__getclasses__()
num_classes = len(classes)
print('num_classes:', num_classes)

batch_size = 20
valid_size = 0.2
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)

test_data = DogBreedsTestset(
    csv_file=os.path.join(data_dir, 'sample_submission.csv'),
    root_dir=os.path.join(data_dir, 'test'),
    transform=transform
)
test_loader = DataLoader(test_data, batch_size=batch_size)

# ================== 4. 模型加载 ==================
model = vgg11(num_classes=num_classes, pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
if train_on_gpu:
    model.cuda()
print(model)

# ================== 5. 损失与优化器 ==================
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================== 6. 训练循环 ==================
n_epochs = 15
for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0.0
    for batch_i, (data, target) in enumerate(train_loader):
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)  # <--- target 已为int型标签
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_i % 20 == 19:
            print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_i + 1, train_loss / 20))
            train_loss = 0.0
    # 验证
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch_i, (data, target) in enumerate(valid_loader):
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()
            if batch_i % 20 == 19:
                print('Validation Loss Batch %d loss: %.6f' %
                      (batch_i + 1, valid_loss / 20))
                valid_loss = 0.0

# ================== 7. 推理与输出 ==================
results = {}
model.eval()
with torch.no_grad():
    for data in test_loader:
        images, titles = data['image'], data['title']
        if train_on_gpu:
            images = images.cuda()
        logits = model(images)
        output = torch.softmax(logits, dim=1)
        for k in range(len(titles)):
            name = titles[k]
            results[name] = output[k].cpu().tolist()

output_df = pd.DataFrame(results).transpose()
inv_map = {v: k for k, v in train_data.getmap().items()}
output_df.rename(columns=inv_map, inplace=True)
output_df = output_df.reset_index()
output_df.rename(columns={'index':'id'}, inplace=True)
output_path = os.path.join(save_dir, 'output.csv')
output_df.to_csv(output_path, index=False)
print(f"已保存提交文件到 {output_path}")
