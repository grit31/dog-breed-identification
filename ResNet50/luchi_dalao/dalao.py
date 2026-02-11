# 导入相关的包
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models
# 使用tqdm库创建进度条
from tqdm import tqdm
import seaborn as sns

# 看看label文件长啥样
breeds_dataframe = pd.read_csv('../input/dog-breed-identification/labels.csv')
breeds_dataframe.head(5)


# 显示条形图
def barw(ax):
    for p in ax.patches:
        val = p.get_width()  # height of the bar
        x = p.get_x() + p.get_width()  # x- position
        y = p.get_y() + p.get_height() / 2  # y-position
        ax.annotate(round(val, 2), (x, y))


# finding top dog
plt.figure(figsize=(15, 30))
ax0 = sns.countplot(y=breeds_dataframe['breed'], order=breeds_dataframe['breed'].value_counts().index)
barw(ax0)
plt.show()

# 把狗的品种排个序
dog_breeds = sorted(list(set(breeds_dataframe['breed'])))
n_breeds = len(dog_breeds)
print(n_breeds)
dog_breeds[:10]

# 把品种转成对应的数字
breed_to_num = dict(zip(dog_breeds, range(n_breeds)))
breed_to_num


# 继承pytorch的dataset，创建自己的数据集
class DogDataset(Dataset):
    def __init__(self, csv_path, file_path, mode="train", transform=None, valid_ratio=0.2,
                 resize_height=224, resize_width=224):
        """
        Args:
            csv_path: csv文件路径
            img_path: 图像文件路径
            mode: 训练模式还是测试模式
            valid_ratio:验证集比例
        """

        # 将图片大小调整为224 * 224
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.file_path = file_path
        self.mode = mode
        self.transform = transform

        # 用pandas读取csv文件
        if csv_path != None:
            self.data_info = pd.read_csv(csv_path)
            # 计算 length
            self.data_len = len(self.data_info.index)
            self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == "train":
            # 第一列包含图像文件的名称
            self.train_image = np.asarray(self.data_info.iloc[0:self.train_len, 0])
            # 第二列是图像的 breed
            self.train_breed = np.asarray(self.data_info.iloc[0:self.train_len, 1])
            self.image_arr = self.train_image
            self.breed_arr = self.train_breed
        elif mode == "valid":
            # 第一列包含图像文件的名称
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])
            # 第二列是图像的 breed
            self.valid_breed = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.breed_arr = self.valid_breed
        elif mode == 'test':
            self.image_arr = [f.split('.')[0] for f in os.listdir(file_path)]

        self.real_len = len(self.image_arr)
        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        # 从image_arr得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像, 拼接出完成的路径
        img_as_img = Image.open(self.file_path + single_image_name + ".jpg")
        # 图像增广
        img_as_img = self.transform(img_as_img)

        if self.mode == "test":
            return img_as_img, single_image_name  # 把图像id也返回
        else:
            # 得到图像的 breed
            breed = self.breed_arr[index]
            # breed转换为对应的数字
            number_breed = breed_to_num[breed]
            return img_as_img, number_breed  # 返回每一个index对应的图片数据和

    def __len__(self):
        return self.real_len

# 下面进行图像增广
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224, # 然后，缩放图像以创建224x224的新图像
        scale=(0.08, 1.0),  # 随机裁剪图像，所得图像为原始面积的0.08～1之间，高宽比在3/4和4/3之间。
        ratio=(3.0/4.0, 4.0/3.0)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    # 随机更改亮度，对比度和饱和度
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4
    ),
    transforms.ToTensor(),
    # 标准化图像的每个通道
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
])

train_path = '/kaggle/input/dog-breed-identification/labels.csv'
train_img_path = '/kaggle/input/dog-breed-identification/train/'
test_img_path = '/kaggle/input/dog-breed-identification/test/'

train_dataset = DogDataset(train_path, train_img_path, mode="train", transform=train_transform)
valid_dataset = DogDataset(train_path, train_img_path, mode='valid', transform=test_transform)
test_dataset = DogDataset(None, test_img_path, mode='test', transform=test_transform)
print(train_dataset)
print(valid_dataset)
print(test_dataset)

batch_size = 128
train_iter = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
)

valid_iter = DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False
)

test_iter = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)

# 看一下是在cpu还是GPU上
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
print(device)

def get_net(device):
    finetune_net = nn.Sequential()
    # 特征提取部分，使用预训练的resnet50
    finetune_net.features = torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT")
    # 定义一个新的输出网络，共有120个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(device)
    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

# 超参数
learning_rate = 1e-4
weight_decay = 1e-3
num_epoch = 50
model_path = '/kaggle/working/pre_res_model.ckpt'

# 初始化模型，并放在指定设备
model = get_net(device)

# 加载训练好的模型
# model.load_state_dict(torch.load(model_path))
# model.device = device

# 分类任务，使用交叉熵cross-entropy作为损失函数
loss_func = nn.CrossEntropyLoss()

# 初始化优化器，可以微调一些超参数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=weight_decay)
# 训练的epochs
n_epochs = num_epoch

# 保存每个epoch的损失和精度
total_train_loss = []
total_train_accs = []
total_valid_loss = []
total_valid_accs = []

best_acc = 0.0
for epoch in range(n_epochs):
    # -------------------------开始训练--------------------------
    # 确保模型处于训练模式
    model.train()
    # These are used to record information in training.
    train_loss = []
    train_accs = []
    # Iterate the training set by batches.
    for batch in tqdm(train_iter):
        # A batch consists of image data and corresponding breed.
        imgs, breeds = batch
        # Forward the data. (Make sure data and model are on the same device.)
        imgs = imgs.to(device)
        breeds = breeds.to(device)
        # 上一步骤中存储在参数中的梯度应首先清除。
        optimizer.zero_grad()
        # 预测
        logits = model(imgs)
        # Calculate the cross-entropy loss.
        loss = loss_func(logits, breeds)
        # 计算梯度，更新参数
        loss.backward()
        optimizer.step()

        # 计算当前批次的精确度
        # logits.argmax(dim=-1) 即求出每张图的类别
        acc = (logits.argmax(dim=-1) == breeds).float().mean()

        # 记录损失和精度
        train_loss.append(loss)
        train_accs.append(acc)
        # for结束
    # 本次训练的平均损失和平均精度
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    # 保存信息
    total_train_loss.append(train_loss.item())
    total_train_accs.append(train_acc.item())

    # 打印信息
    print(f"[Train | {epoch + 1:03d}/{n_epochs:03d}] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules
    # like dropout are disabled and work normally.
    model.eval()
    # 记录验证过程的信息
    valid_loss = []
    valid_accs = []

    # 按照批次迭代验证
    for batch in tqdm(valid_iter):
        imgs, breeds = batch
        imgs = imgs.to(device)
        breeds = breeds.to(device)
        # 验证过程不用计算梯度
        with torch.no_grad():
            logits = model(imgs)

        # 计算损失，但是没有梯度
        loss = loss_func(logits, breeds)

        # 计算当前批次的精度
        acc = (logits.argmax(dim=-1) == breeds).float().mean()
        # 记录损失和精度
        valid_loss.append(loss)
        valid_accs.append(acc)

    # 整个验证过程的平均损失和精度
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    total_valid_loss.append(valid_loss.item())
    total_valid_accs.append(valid_acc.item())

    # 打印信息
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

# 保存模型
torch.save(model.state_dict(), model_path)
print('模型保存成功，目录为：{}'.format(model_path))

print(total_train_loss)
print(total_train_accs)
print(total_valid_loss)
print(total_valid_accs)

# 画出损失函数和精度图
# 设置Seaborn的风格
sns.set(style="darkgrid")
epochs = np.arange(1, num_epoch+1, 1)
# 创建一个画布
plt.figure(figsize=(12, 12))

# 绘制损失图
plt.subplot(2, 1, 1)
sns.lineplot(x=epochs, y=total_train_loss, label='Train Loss')
sns.lineplot(x=epochs, y=total_valid_loss, label='Valid Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制精度图
plt.subplot(2, 1, 2)
sns.lineplot(x=epochs, y=total_train_accs, label='Train Accuracy')
sns.lineplot(x=epochs, y=total_valid_accs, label='Valid Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 显示图表
# plt.tight_layout()
plt.show()

# 保存文件目录
saveFileName = '/kaggle/working/submission.csv'
model =  get_net(device)
# 加载训练好的模型
model.load_state_dict(torch.load(model_path))

test_path = "/kaggle/input/dog-breed-identification/test"
predict = pd.DataFrame(columns=["id",*dog_breeds])
ids = sorted([f.split('.')[0] for f in os.listdir(test_path)])
# ids

## predict
# 开启eval模式,评估模式
model.eval()

preds = [] # 保存预测结果
ids = [] # 保存每张图片的id
# 按照批次迭代
for batch in tqdm(test_iter):
    imgs, imgs_id = batch
    with torch.no_grad():
        # 使用softmax对结果处理
        output = torch.nn.functional.softmax(model(imgs.to(device)), dim=1)
        preds.extend(output.cpu().numpy())
        ids.extend(imgs_id)

print(preds[:2])
print(ids[:2])

ids[0] + ',' + ','.join([str(n) for n in preds[0]])

with open('/kaggle/working/submission.csv', 'w', encoding='utf-8') as f:
    f.write('id,' + ','.join(dog_breeds) + '\n')
    for i, output in zip(ids, preds):
        f.write(i + ',' + ','.join([str(n) for n in output]) + '\n')