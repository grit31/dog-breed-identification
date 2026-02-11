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
import matplotlib.pyplot as plt

# =============== 1.1 VGG 配置表，可选结构 ===============
cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# =============== 1.2 VGG 网络实现 ===============
class VGG(nn.Module):
    def __init__(self, features, num_classes=120):
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

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def build_vgg(cfg_name='VGG11', num_classes=120):
    return VGG(make_layers(cfgs[cfg_name]), num_classes=num_classes)

# =============== 2. 路径配置 ===============
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
indices = list(range(data_size))
split = int(np.round(0.2*data_size,0))
training_indices = indices[split:]
validation_indices = indices[:split]
train_sampler = SubsetRandomSampler(training_indices)
valid_sampler = SubsetRandomSampler(validation_indices)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
valid_loader = DataLoader(train_dataset, batch_size=32, sampler=valid_sampler)

# =============== 5. 模型与优化器 ===============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_vgg('VGG11', num_classes=120).to(device)
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
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1} Training')
        for batch_idx, d in pbar:
            data = d['images'].to(device)
            target = d['labels'].to(device)
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
                pbar.set_postfix(loss=training_loss, acc=100*correct/total)
        train_losses.append(training_loss)
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, d in enumerate(valid_loader):
                data = d['images'].to(device)
                target = d['labels'].to(device)
                output = model(data)
                loss = criteria(output, target)
                validation_loss = validation_loss + ((1) / (batch_idx + 1) * (loss.item() - validation_loss))
                pred = output.data.max(1, keepdim=True)[1]
                correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
                total += data.size(0)
        val_losses.append(validation_loss)
        print(f"Epoch {epoch+1}: Train Loss={training_loss:.4f} | Val Loss={validation_loss:.4f} | Val Acc={100*correct/total:.2f}%")
        torch.save(model.state_dict(), os.path.join(save_dir, f'vgg11_epoch{epoch+1}.pth'))
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VGG11 Training & Validation Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vgg11_loss_curve.png'))
    print('Loss曲线已保存到:', os.path.join(save_dir, 'vgg11_loss_curve.png'))
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
    data = d['images'].to(device)
    with torch.no_grad():
        output = model(data)
        output = torch.softmax(output, dim=1).cpu().detach().numpy()
        predict.append(list(output[0]))
for i in tqdm(range(len(predict))):
    sample.iloc[i, 1:] = predict[i]
sample.to_csv(os.path.join(save_dir, 'sample_submission.csv'), index=False)
print('预测结果保存到:', os.path.join(save_dir, 'sample_submission.csv'))
