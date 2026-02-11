import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt  # 新增

# =============== 1. 手动搭建ResNet残差结构 ===============
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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
        super().__init__()
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

def build_resnet18(num_classes=120):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# =============== 2. 路径前缀配置 ===============
data_dir = r'D:\homework\Course_Design_of_artificial_intelligence\third\dog-breed-identification'
save_dir = r'D:\homework\Course_Design_of_artificial_intelligence\third\ResNet\kaggle_dalao\sam_dalao_before'

# =============== 3. 标签处理 ===============
le = LabelEncoder()
df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
df['breed'] = le.fit_transform(df['breed'])

# =============== 4. 数据集定义 ===============
class DogDataset(Dataset):
    def __init__(self, csv, transform):
        self.data = csv
        self.transform = transform
        self.labels = torch.eye(120)[self.data['breed']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(data_dir, 'train', self.data.loc[idx]['id'] + '.jpg')
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.data.loc[idx]['breed'])
        return {'images': image, 'labels': label}

simple_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = DogDataset(df, simple_transform)
data_size = len(train_dataset)
indicies = list(range(data_size))
split = int(np.round(0.2*data_size,0))
training_indicies = indicies[split:]
validation_indices = indicies[:split]
train_sampler = SubsetRandomSampler(training_indicies)
valid_sampler = SubsetRandomSampler(validation_indices)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
valid_loader = DataLoader(train_dataset, batch_size=32, sampler=valid_sampler)

# =============== 5. 模型加载与优化器 ===============
model = build_resnet18(num_classes=120)
# model.load_state_dict(torch.load(os.path.join(data_dir, 'resnet18.pth')))
model = model.cuda()
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =============== 6. 训练过程 + Loss 曲线绘制 ===============
def fit(epochs, model, optimizer, criteria):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        training_loss = 0.0
        validation_loss = 0.0
        correct = 0
        total = 0
        print('{}/{} Epochs'.format(epoch + 1, epochs))
        model.train()
        for batch_idx, d in enumerate(train_loader):
            data = d['images'].cuda()
            #target = d['labels'].cuda()
            target = d['labels'].cuda().long()   # <--- 只加 .long() 即可
            optimizer.zero_grad()
            output = model(data)
            loss = criteria(output, target)
            loss.backward()
            optimizer.step()
            training_loss = training_loss + (1 / (batch_idx + 1) * (loss.item() - training_loss))
            pred = output.data.max(1, keepdim=True)[1]
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)
            if batch_idx % 20 == 0:
                print('Training Loss: {:.4f}, Accuracy: {:.2f}%'.format(training_loss, 100 * correct / total))
        train_losses.append(training_loss)
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, d in enumerate(valid_loader):
                data = d['images'].cuda()
                #target = d['labels'].cuda()
                target = d['labels'].cuda().long()   # <--- 只加 .long() 即可)
                output = model(data)
                loss = criteria(output, target)
                validation_loss = validation_loss + ((1) / (batch_idx + 1) * (loss.item() - validation_loss))
                pred = output.data.max(1, keepdim=True)[1]
                correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
                total += data.size(0)
                if batch_idx % 20 == 0:
                    print('Validation Loss: {:.4f}, Accuracy: {:.2f}%'.format(validation_loss, 100 * correct / total))
        val_losses.append(validation_loss)
        # 可选：保存模型
        torch.save(model.state_dict(), os.path.join(save_dir, f'resnet18_epoch{epoch+1}.pth'))

    # ==== 新增：绘制loss曲线 ====
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ResNet18 Training & Validation Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    print('Loss曲线已保存到:', os.path.join(save_dir, 'loss_curve.png'))
    plt.close()
    return model

fit(30, model, optimizer, criteria)

# =============== 7. 预测并生成Kaggle提交 ===============
sample = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

class Prediction(Dataset):
    def __init__(self, csv, transform):
        self.data = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(data_dir, 'test', self.data.loc[idx]['id'] + '.jpg')
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return {'images': image}

test_dataset = Prediction(os.path.join(data_dir, 'sample_submission.csv'), simple_transform)
test_loader = DataLoader(test_dataset)

model.eval()
predict = []
for batch_idx, d in enumerate(test_loader):
    data = d['images'].cuda()
    with torch.no_grad():
        output = model(data)
        output = torch.softmax(output, dim=1).cpu().detach().numpy()
        predict.append(list(output[0]))

for i in tqdm(range(len(predict))):
    sample.iloc[i, 1:] = predict[i]
sample.to_csv(os.path.join(save_dir, 'sample_submission.csv'), index=False)
print('预测结果保存到:', os.path.join(save_dir, 'sample_submission.csv'))