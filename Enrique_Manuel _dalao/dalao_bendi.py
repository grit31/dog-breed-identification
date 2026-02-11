# import os
# import pandas as pd
# import numpy as np
# from PIL import Image
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# import torch as T
# from torch import nn
# from torch.utils.data import DataLoader, Dataset
# from torchvision import models, transforms
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# device = T.device('cuda' if T.cuda.is_available() else 'cpu')
#
# # 路径设置（自行调整为你本地的文件夹）
# DATA_DIR = r'D:\homework\Course_Design_of_artificial_intelligence\third\dog-breed-identification'
# TRAIN_IMG_PATH = os.path.join(DATA_DIR, 'train')
# TEST_IMG_PATH = os.path.join(DATA_DIR, 'test')
# LABELS_PATH = os.path.join(DATA_DIR, 'labels.csv')
#
# # 1. 读入数据
# train_data = pd.read_csv(LABELS_PATH)
# print(f"Train dataset shape: {train_data.shape}")
# print(train_data.head())
#
# # 2. 生成test数据
# test_files = [f for f in os.listdir(TEST_IMG_PATH) if f.endswith('.jpg')]
# test_data = pd.DataFrame({'id': [os.path.splitext(f)[0] for f in test_files]})
# print(f"Test dataset shape: {test_data.shape}")
# print(test_data.head())
#
# # 3. 标签编码
# le = LabelEncoder()
# train_data['breed'] = le.fit_transform(train_data['breed'])
#
# # 4. 数据集类
# class Dog_Breed_Dataset(Dataset):
#     def __init__(self, df: pd.DataFrame, img_base_path: str, split: str, transforms=None):
#         self.df = df
#         self.img_base_path = img_base_path
#         self.split = split
#         self.transforms = transforms
#
#     def __getitem__(self, index):
#         img_path = os.path.join(self.img_base_path, self.df.loc[index, 'id'] + '.jpg')
#         img = Image.open(img_path).convert('RGB')
#         if self.transforms:
#             img = self.transforms(img)
#         if self.split != 'test':
#             y = self.df.loc[index, 'breed']
#             return img, y
#         else:
#             return img
#
#     def __len__(self):
#         return len(self.df)
#
# # 5. 数据增强
# train_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(p=0.2),
#     transforms.RandomVerticalFlip(p=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# test_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# # 6. 划分数据集
# train, val = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['breed'])
# train = train.reset_index(drop=True)
# val = val.reset_index(drop=True)
#
# train_dataset = Dog_Breed_Dataset(train, TRAIN_IMG_PATH, 'train', train_transforms)
# validation_dataset = Dog_Breed_Dataset(val, TRAIN_IMG_PATH, 'val', test_transforms)
# test_dataset = Dog_Breed_Dataset(test_data, TEST_IMG_PATH, 'test', test_transforms)
#
# train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
# validation_dl = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0)
# test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
#
# print(f"Train data length: {len(train_dl.dataset)}, Validation data length: {len(validation_dl.dataset)}, Test data length: {len(test_dl.dataset)}")
#
# # 7. 投票方法
# def voter(pred1, pred2, pred3, pred4, pred5):
#     final_predictions = []
#     for i in range(pred1.size(0)):
#         preds = [pred1[i].item(), pred2[i].item(), pred3[i].item(), pred4[i].item(), pred5[i].item()]
#         counts = pd.Series(preds).value_counts()
#         pred = counts.index[0]
#         final_predictions.append(pred)
#     return T.tensor(final_predictions)
#
# # 8. 模型集成定义
# inception = models.inception_v3(pretrained=True, aux_logits=True)
# resnet50 = models.resnet50(pretrained=True)
# # 保证特征维度一致：flatten到2048
# class Model(nn.Module):
#     def __init__(self, inception_model, resnet50_model, num_classes=120):
#         super().__init__()
#         self.inception_model = nn.Sequential(
#             inception_model.Conv2d_1a_3x3,
#             inception_model.Conv2d_2a_3x3,
#             inception_model.Conv2d_2b_3x3,
#             inception_model.maxpool1,
#             inception_model.Conv2d_3b_1x1,
#             inception_model.Conv2d_4a_3x3,
#             inception_model.maxpool2,
#             inception_model.Mixed_5b,
#             inception_model.Mixed_5c,
#             inception_model.Mixed_5d,
#             inception_model.Mixed_6a,
#             inception_model.Mixed_6b,
#             inception_model.Mixed_6c,
#             inception_model.Mixed_6d,
#             inception_model.Mixed_6e,
#             inception_model.Mixed_7a,
#             inception_model.Mixed_7b,
#             inception_model.Mixed_7c,
#             inception_model.avgpool
#         )
#         self.resnet50_model = nn.Sequential(
#             resnet50_model.conv1,
#             resnet50_model.bn1,
#             resnet50_model.relu,
#             resnet50_model.maxpool,
#             resnet50_model.layer1,
#             resnet50_model.layer2,
#             resnet50_model.layer3,
#             resnet50_model.layer4,
#             resnet50_model.avgpool
#         )
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(2048 * 2, num_classes)  # 两个模型拼接
#         self.num_classes = num_classes
#
#         # 优化器
#         self.optim = T.optim.SGD(self.fc.parameters(), lr=0.005, momentum=0.9)
#         self.optim_resnet = T.optim.Adam(self.resnet50_model.parameters(), lr=0.0001)
#         self.optim_inception = T.optim.Adam(self.inception_model.parameters(), lr=0.0001)
#         self.criterion = nn.CrossEntropyLoss()
#         self.scheduler = ReduceLROnPlateau(self.optim, mode='min', factor=0.1, patience=5)
#         self.to(device)
#
#     def forward(self, x):
#         X1 = self.inception_model(x)
#         X2 = self.resnet50_model(x)
#         X1 = X1.view(X1.size(0), -1)
#         X2 = X2.view(X2.size(0), -1)
#         X = T.cat([X1, X2], dim=1)
#         X = self.dropout(X)
#         P1 = self.fc(X)
#         P2 = self.fc(X)
#         P3 = self.fc(X)
#         P4 = self.fc(X)
#         P5 = self.fc(X)
#         return [P1, P2, P3, P4, P5]
#
#     def get_weights(self):
#         return self.state_dict()
#
#     def load_weights(self, weights):
#         self.load_state_dict(weights)
#
# # 9. 训练函数
# def train_model(train_dl, val_dl, model, epochs=6):
#     train_acc_history = []
#     val_acc_history = []
#     train_loss_history = []
#     val_loss_history = []
#     best_val_loss = 1_000_000.0
#     weights = model.get_weights()
#
#     for epoch in range(epochs):
#         print("=" * 20, "Epoch: ", str(epoch), "=" * 20)
#         train_correct_pred, val_correct_pred = 0, 0
#         train_loss, val_loss = 0.0, 0.0
#
#         model.train()
#         for x, y in train_dl:
#             x, y = x.to(device), y.long().to(device)
#             model.optim.zero_grad()
#             model.optim_resnet.zero_grad()
#             model.optim_inception.zero_grad()
#             preds = model(x)
#             loss = sum(model.criterion(p, y) for p in preds) / len(preds)
#             loss.backward()
#             model.optim.step()
#             model.optim_resnet.step()
#             model.optim_inception.step()
#             train_loss += loss.item()
#             preds_label = [T.argmax(T.softmax(p, dim=1), dim=1) for p in preds]
#             final_pred = voter(*preds_label)
#             train_correct_pred += (final_pred.to(y.device) == y).sum().item()
#
#         train_acc = train_correct_pred / len(train_dl.dataset)
#         train_acc_history.append(train_acc)
#         train_loss_history.append(train_loss)
#
#         model.eval()
#         with T.no_grad():
#             for x, y in val_dl:
#                 x, y = x.to(device), y.long().to(device)
#                 preds = model(x)
#                 loss = sum(model.criterion(p, y) for p in preds) / len(preds)
#                 val_loss += loss.item()
#                 preds_label = [T.argmax(T.softmax(p, dim=1), dim=1) for p in preds]
#                 final_pred = voter(*preds_label)
#                 val_correct_pred += (final_pred.to(y.device) == y).sum().item()
#         val_acc = val_correct_pred / len(val_dl.dataset)
#         val_acc_history.append(val_acc)
#         val_loss_history.append(val_loss)
#         model.scheduler.step(val_loss)
#         if best_val_loss > val_loss:
#             best_val_loss = val_loss
#             weights = model.get_weights()
#         print(f"Train acc: {train_acc:.4f} | Train Loss: {train_loss:.4f} | Validation acc: {val_acc:.4f} | Validation Loss: {val_loss:.4f}")
#     model.load_weights(weights)
#
#     # 测试集推理
#     model.eval()
#     prob_preds = []
#     with T.no_grad():
#         for x in test_dl:
#             x = x.to(device)
#             preds = model(x)
#             prob_pred = sum(T.softmax(p, dim=1) for p in preds) / len(preds)
#             prob_preds.append(prob_pred.cpu().numpy())
#     prob_preds = np.vstack(prob_preds)
#     prob_preds_df = pd.DataFrame(prob_preds)
#     return [train_acc_history, train_loss_history, val_acc_history, val_loss_history], prob_preds_df
#
# # 10. 训练与推理
# model = Model(inception, resnet50, num_classes=len(le.classes_))
# history, test_preds = train_model(train_dl, validation_dl, model, epochs=1)
#
# # 11. 可视化
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(history[0], label="Training accuracy")
# plt.plot(history[2], label="Validation accuracy")
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.legend()
# plt.subplot(1,2,2)
# plt.plot(history[1], label="Training Loss")
# plt.plot(history[3], label="Validation Loss")
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.legend()
# plt.suptitle("Training and Validation Results of Model")
# plt.show()
#
# # 12. 生成提交文件
# num_classes = np.array(test_preds.columns)
# num_classes = le.inverse_transform(num_classes.astype(int))
# test_preds.columns = list(num_classes)
# test_preds = test_preds.reset_index(drop=True)
# ids = test_data['id']
# test_preds = pd.concat([ids, test_preds], axis=1)
# test_preds.to_csv('submission.csv', index=None)
# print("submission.csv saved.")


import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch as T
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm  # 新增
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = T.device('cuda' if T.cuda.is_available() else 'cpu')

# 路径设置
DATA_DIR = r'/kaggle/input/dog-breed-identification'
TRAIN_IMG_PATH = os.path.join(DATA_DIR, 'train')
TEST_IMG_PATH = os.path.join(DATA_DIR, 'test')
LABELS_PATH = os.path.join(DATA_DIR, 'labels.csv')

# 1. 读入数据
train_data = pd.read_csv(LABELS_PATH)
print(f"Train dataset shape: {train_data.shape}")
print(train_data.head())

# 2. 生成test数据
test_files = [f for f in os.listdir(TEST_IMG_PATH) if f.endswith('.jpg')]
test_data = pd.DataFrame({'id': [os.path.splitext(f)[0] for f in test_files]})
print(f"Test dataset shape: {test_data.shape}")
print(test_data.head())

# 3. 标签编码
le = LabelEncoder()
train_data['breed'] = le.fit_transform(train_data['breed'])

# 4. 数据集类
class Dog_Breed_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_base_path: str, split: str, transforms=None):
        self.df = df
        self.img_base_path = img_base_path
        self.split = split
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = os.path.join(self.img_base_path, self.df.loc[index, 'id'] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        if self.split != 'test':
            y = self.df.loc[index, 'breed']
            return img, y
        else:
            return img

    def __len__(self):
        return len(self.df)

# 5. 数据增强
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 6. 划分数据集
train, val = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['breed'])
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)

train_dataset = Dog_Breed_Dataset(train, TRAIN_IMG_PATH, 'train', train_transforms)
validation_dataset = Dog_Breed_Dataset(val, TRAIN_IMG_PATH, 'val', test_transforms)
test_dataset = Dog_Breed_Dataset(test_data, TEST_IMG_PATH, 'test', test_transforms)

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
validation_dl = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Train data length: {len(train_dl.dataset)}, Validation data length: {len(validation_dl.dataset)}, Test data length: {len(test_dl.dataset)}")

# 7. 投票方法
def voter(pred1, pred2, pred3, pred4, pred5):
    final_predictions = []
    for i in range(pred1.size(0)):
        preds = [pred1[i].item(), pred2[i].item(), pred3[i].item(), pred4[i].item(), pred5[i].item()]
        counts = pd.Series(preds).value_counts()
        pred = counts.index[0]
        final_predictions.append(pred)
    return T.tensor(final_predictions)

# 8. 模型集成定义
inception = models.inception_v3(pretrained=True, aux_logits=True)
resnet50 = models.resnet50(pretrained=True)

class Model(nn.Module):
    def __init__(self, inception_model, resnet50_model, num_classes=120):
        super().__init__()
        self.inception_model = nn.Sequential(
            inception_model.Conv2d_1a_3x3,
            inception_model.Conv2d_2a_3x3,
            inception_model.Conv2d_2b_3x3,
            inception_model.maxpool1,
            inception_model.Conv2d_3b_1x1,
            inception_model.Conv2d_4a_3x3,
            inception_model.maxpool2,
            inception_model.Mixed_5b,
            inception_model.Mixed_5c,
            inception_model.Mixed_5d,
            inception_model.Mixed_6a,
            inception_model.Mixed_6b,
            inception_model.Mixed_6c,
            inception_model.Mixed_6d,
            inception_model.Mixed_6e,
            inception_model.Mixed_7a,
            inception_model.Mixed_7b,
            inception_model.Mixed_7c,
            inception_model.avgpool
        )
        self.resnet50_model = nn.Sequential(
            resnet50_model.conv1,
            resnet50_model.bn1,
            resnet50_model.relu,
            resnet50_model.maxpool,
            resnet50_model.layer1,
            resnet50_model.layer2,
            resnet50_model.layer3,
            resnet50_model.layer4,
            resnet50_model.avgpool
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(2048 * 2, num_classes)  # 两个模型拼接
        self.num_classes = num_classes

        self.optim = T.optim.SGD(self.fc.parameters(), lr=0.005, momentum=0.9)
        self.optim_resnet = T.optim.Adam(self.resnet50_model.parameters(), lr=0.0001)
        self.optim_inception = T.optim.Adam(self.inception_model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(self.optim, mode='min', factor=0.1, patience=5)
        self.to(device)

    def forward(self, x):
        X1 = self.inception_model(x)
        X2 = self.resnet50_model(x)
        X1 = X1.view(X1.size(0), -1)
        X2 = X2.view(X2.size(0), -1)
        X = T.cat([X1, X2], dim=1)
        X = self.dropout(X)
        P1 = self.fc(X)
        P2 = self.fc(X)
        P3 = self.fc(X)
        P4 = self.fc(X)
        P5 = self.fc(X)
        return [P1, P2, P3, P4, P5]

    def get_weights(self):
        return self.state_dict()

    def load_weights(self, weights):
        self.load_state_dict(weights)

# 9. 训练函数（添加 tqdm）
def train_model(train_dl, val_dl, model, epochs=6):
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    best_val_loss = 1_000_000.0
    weights = model.get_weights()

    for epoch in range(epochs):
        print("=" * 20, "Epoch: ", str(epoch), "=" * 20)
        train_correct_pred, val_correct_pred = 0, 0
        train_loss, val_loss = 0.0, 0.0

        # --- Train ---
        model.train()
        tbar = tqdm(train_dl, desc="Training", leave=False)
        for x, y in tbar:
            x, y = x.to(device), y.long().to(device)
            model.optim.zero_grad()
            model.optim_resnet.zero_grad()
            model.optim_inception.zero_grad()
            preds = model(x)
            loss = sum(model.criterion(p, y) for p in preds) / len(preds)
            loss.backward()
            model.optim.step()
            model.optim_resnet.step()
            model.optim_inception.step()
            train_loss += loss.item()
            preds_label = [T.argmax(T.softmax(p, dim=1), dim=1) for p in preds]
            final_pred = voter(*preds_label)
            train_correct_pred += (final_pred.to(y.device) == y).sum().item()
            tbar.set_postfix({'batch_loss': loss.item()})

        train_acc = train_correct_pred / len(train_dl.dataset)
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)

        # --- Validation ---
        model.eval()
        tbar = tqdm(val_dl, desc="Validation", leave=False)
        with T.no_grad():
            for x, y in tbar:
                x, y = x.to(device), y.long().to(device)
                preds = model(x)
                loss = sum(model.criterion(p, y) for p in preds) / len(preds)
                val_loss += loss.item()
                preds_label = [T.argmax(T.softmax(p, dim=1), dim=1) for p in preds]
                final_pred = voter(*preds_label)
                val_correct_pred += (final_pred.to(y.device) == y).sum().item()
                tbar.set_postfix({'batch_loss': loss.item()})

        val_acc = val_correct_pred / len(val_dl.dataset)
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)
        model.scheduler.step(val_loss)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            weights = model.get_weights()
        print(f"Train acc: {train_acc:.4f} | Train Loss: {train_loss:.4f} | Validation acc: {val_acc:.4f} | Validation Loss: {val_loss:.4f}")

    model.load_weights(weights)

    # --- 测试集推理 ---
    model.eval()
    prob_preds = []
    tbar = tqdm(test_dl, desc="Inference", leave=False)
    with T.no_grad():
        for x in tbar:
            x = x.to(device)
            preds = model(x)
            prob_pred = sum(T.softmax(p, dim=1) for p in preds) / len(preds)
            prob_preds.append(prob_pred.cpu().numpy())
    prob_preds = np.vstack(prob_preds)
    prob_preds_df = pd.DataFrame(prob_preds)
    return [train_acc_history, train_loss_history, val_acc_history, val_loss_history], prob_preds_df

# 10. 训练与推理
model = Model(inception, resnet50, num_classes=len(le.classes_))
history, test_preds = train_model(train_dl, validation_dl, model, epochs=1)

# 11. 可视化
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history[0], label="Training accuracy")
plt.plot(history[2], label="Validation accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.subplot(1,2,2)
plt.plot(history[1], label="Training Loss")
plt.plot(history[3], label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.suptitle("Training and Validation Results of Model")
plt.show()

# 12. 生成提交文件
num_classes = np.array(test_preds.columns)
num_classes = le.inverse_transform(num_classes.astype(int))
test_preds.columns = list(num_classes)
test_preds = test_preds.reset_index(drop=True)
ids = test_data['id']
test_preds = pd.concat([ids, test_preds], axis=1)
test_preds.to_csv('submission.csv', index=None)
print("submission.csv saved.")
