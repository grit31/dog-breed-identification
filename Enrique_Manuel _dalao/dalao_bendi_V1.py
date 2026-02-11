# import os
# import time
# import pandas as pd
# import numpy as np
# from PIL import Image
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# import torch as T
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, Dataset
# from torchvision import models, transforms
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# import psutil
# from thop import profile
# from sklearn.metrics import accuracy_score, top_k_accuracy_score
#
# # =================== 路径与设备 =====================
# DATA_DIR = r'/kaggle/input/dog-breed-identification'
# SAVE_DIR = './ensemble_metrics_output'
# os.makedirs(SAVE_DIR, exist_ok=True)
# TRAIN_IMG_PATH = os.path.join(DATA_DIR, 'train')
# TEST_IMG_PATH = os.path.join(DATA_DIR, 'test')
# LABELS_PATH = os.path.join(DATA_DIR, 'labels.csv')
# device = T.device('cuda' if T.cuda.is_available() else 'cpu')
#
# # =================== 1. 数据读取 =====================
# train_data = pd.read_csv(LABELS_PATH)
# test_files = [f for f in os.listdir(TEST_IMG_PATH) if f.endswith('.jpg')]
# test_data = pd.DataFrame({'id': [os.path.splitext(f)[0] for f in test_files]})
#
# # 标签编码
# le = LabelEncoder()
# train_data['breed'] = le.fit_transform(train_data['breed'])
#
# # =================== 2. 数据集类 =====================
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
# # =================== 3. 增强与数据加载 =====================
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
# train, val = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['breed'])
# train = train.reset_index(drop=True)
# val = val.reset_index(drop=True)
# train_dataset = Dog_Breed_Dataset(train, TRAIN_IMG_PATH, 'train', train_transforms)
# validation_dataset = Dog_Breed_Dataset(val, TRAIN_IMG_PATH, 'val', test_transforms)
# test_dataset = Dog_Breed_Dataset(test_data, TEST_IMG_PATH, 'test', test_transforms)
# train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
# validation_dl = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0)
# test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
#
# # =================== 4. 投票法 =====================
# def voter(pred1, pred2, pred3, pred4, pred5):
#     final_predictions = []
#     for i in range(pred1.size(0)):
#         preds = [pred1[i].item(), pred2[i].item(), pred3[i].item(), pred4[i].item(), pred5[i].item()]
#         counts = pd.Series(preds).value_counts()
#         pred = counts.index[0]
#         final_predictions.append(pred)
#     return T.tensor(final_predictions)
#
# # =================== 5. 集成模型定义 =====================
# inception = models.inception_v3(pretrained=True, aux_logits=True)
# resnet50 = models.resnet50(pretrained=True)
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
#         self.fc = nn.Linear(2048 * 2, num_classes)
#         self.num_classes = num_classes
#
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
#     def load_weights(self, weights):
#         self.load_state_dict(weights)
#
# # =================== 6. 模型统计 =====================
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# def model_size_mb(model):
#     param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
#     buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
#     size_all_mb = (param_size + buffer_size) / 1024 ** 2
#     return size_all_mb
# def get_flops(model, input_size=(1, 3, 224, 224)):
#     x = T.randn(input_size).to(device)
#     macs, params = profile(model, inputs=(x,), verbose=False)
#     flops = 2 * macs
#     return flops, params
#
# # =================== 7. 训练主循环 =====================
# def train_model(train_dl, val_dl, model, epochs=6):
#     total_train_loss, total_train_accs = [], []
#     total_valid_loss, total_valid_accs = [], []
#     train_top5s, valid_top5s = [], []
#     best_val_loss = float('inf')
#     best_top1, best_top5 = 0, 0
#     best_epoch = 0
#     weights = model.get_weights()
#     start_train_time = time.time()
#
#     for epoch in range(epochs):
#         model.train()
#         train_loss, train_accs, train_labels, train_preds = [], [], [], []
#         for x, y in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
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
#             acc = (T.argmax(T.softmax(preds[0], dim=1), dim=1) == y).float().mean()
#             train_loss.append(loss.item() * x.size(0))
#             train_accs.append(acc.item() * x.size(0))
#             train_labels.extend(y.cpu().numpy())
#             train_preds.extend(preds[0].detach().cpu().numpy())
#         n_train = len(train_dl.dataset)
#         avg_train_loss = np.sum(train_loss) / n_train
#         avg_train_acc = np.sum(train_accs) / n_train
#         train_top5 = top_k_accuracy_score(train_labels, np.vstack(train_preds), k=5)
#         total_train_loss.append(avg_train_loss)
#         total_train_accs.append(avg_train_acc)
#         train_top5s.append(train_top5)
#
#         # 验证
#         model.eval()
#         valid_loss, valid_accs, valid_labels, valid_preds = [], [], [], []
#         for x, y in tqdm(val_dl, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
#             x, y = x.to(device), y.long().to(device)
#             with T.no_grad():
#                 preds = model(x)
#                 loss = sum(model.criterion(p, y) for p in preds) / len(preds)
#             acc = (T.argmax(T.softmax(preds[0], dim=1), dim=1) == y).float().mean()
#             valid_loss.append(loss.item() * x.size(0))
#             valid_accs.append(acc.item() * x.size(0))
#             valid_labels.extend(y.cpu().numpy())
#             valid_preds.extend(preds[0].detach().cpu().numpy())
#         n_valid = len(val_dl.dataset)
#         avg_valid_loss = np.sum(valid_loss) / n_valid
#         avg_valid_acc = np.sum(valid_accs) / n_valid
#         valid_top5 = top_k_accuracy_score(valid_labels, np.vstack(valid_preds), k=5)
#         total_valid_loss.append(avg_valid_loss)
#         total_valid_accs.append(avg_valid_acc)
#         valid_top5s.append(valid_top5)
#
#         print(f"[Train | {epoch+1:03d}/{epochs}] loss = {avg_train_loss:.5f}, acc = {avg_train_acc:.5f}, top5 = {train_top5:.5f}")
#         print(f"[Valid | {epoch+1:03d}/{epochs}] loss = {avg_valid_loss:.5f}, acc = {avg_valid_acc:.5f}, top5 = {valid_top5:.5f}")
#
#         if avg_valid_loss < best_val_loss:
#             best_val_loss = avg_valid_loss
#             best_top1 = avg_valid_acc
#             best_top5 = valid_top5
#             best_epoch = epoch + 1
#             weights = model.get_weights()
#             T.save(weights, os.path.join(SAVE_DIR, 'best_ensemble.pth'))
#             print(f">>> Model saved at epoch {best_epoch}, val_loss = {best_val_loss:.5f}")
#
#     total_train_time = time.time() - start_train_time
#     model.load_weights(weights)
#     print(f"\n训练收敛到最优Val Loss的Epoch: {best_epoch}, 总训练用时: {total_train_time/60:.2f} 分钟")
#     print(f"最优Top-1: {best_top1:.4f}, 最优Top-5: {best_top5:.4f}")
#
#     # 返回历史指标
#     history = {
#         'train_loss': total_train_loss,
#         'val_loss': total_valid_loss,
#         'val_acc': total_valid_accs,
#         'val_top5': valid_top5s
#     }
#     return history, total_train_time
#
# # =================== 8. 可视化 =====================
# def plot_metrics(history, save_dir, suffix="ensemble"):
#     plt.figure()
#     plt.plot(history['train_loss'], label='train_loss')
#     plt.plot(history['val_loss'], label='val_loss')
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.title("Loss Curve")
#     plt.savefig(os.path.join(save_dir, f"loss_curve_{suffix}.png"))
#     plt.close()
#
#     plt.figure()
#     plt.plot(history['val_acc'], label='val_acc')
#     plt.plot(history['val_top5'], label='val_top5')
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.title("Accuracy Curve")
#     plt.savefig(os.path.join(save_dir, f"acc_curve_{suffix}.png"))
#     plt.close()
#
# # =================== 9. 训练与保存 =====================
# model = Model(inception, resnet50, num_classes=len(le.classes_))
# print("\n====== 模型参数统计 ======")
# print(f"参数量: {count_parameters(model)/1e6:.2f} M")
# print(f"模型大小: {model_size_mb(model):.2f} MB")
# flops, _ = get_flops(model, input_size=(1, 3, 224, 224))
# print(f"FLOPs: {flops / 1e9:.2f} GFLOPs/图像\n")
#
# history, train_time = train_model(train_dl, validation_dl, model, epochs=6)
# plot_metrics(history, SAVE_DIR, suffix="ensemble")
#
# # =================== 10. 验证集指标（最终） =====================
# model.load_weights(torch.load(os.path.join(SAVE_DIR, 'best_ensemble.pth')))
# model.eval()
# y_true, y_pred, y_pred_logits = [], [], []
# for batch in tqdm(validation_dl, desc="Final Valid Eval"):
#     imgs, breeds = batch
#     imgs = imgs.to(device)
#     breeds = breeds.to(device)
#     with T.no_grad():
#         preds = model(imgs)
#         out = preds[0]
#         pred_labels = out.argmax(dim=1)
#         y_true.extend(breeds.cpu().numpy())
#         y_pred.extend(pred_labels.cpu().numpy())
#         y_pred_logits.extend(out.cpu().numpy())
# val_top1_acc = accuracy_score(y_true, y_pred)
# val_top5_acc = top_k_accuracy_score(y_true, np.vstack(y_pred_logits), k=5)
# print(f"\nFinal Validation Top-1 Acc: {val_top1_acc:.4f}, Top-5 Acc: {val_top5_acc:.4f}")
#
# # =================== 11. 推理速度与内存占用 =====================
# def test_speed(model, dataloader, device):
#     model.eval()
#     total_time = 0
#     n = 0
#     with T.no_grad():
#         for batch in tqdm(dataloader, desc="SpeedTest"):
#             if isinstance(batch, (list, tuple)):
#                 imgs = batch[0]
#             else:
#                 imgs = batch
#             imgs = imgs.to(device)
#             batch_size = imgs.size(0)
#             if device.type == "cuda":
#                 T.cuda.synchronize()
#             start = time.time()
#             _ = model(imgs)
#             if device.type == "cuda":
#                 T.cuda.synchronize()
#             total_time += (time.time() - start)
#             n += batch_size
#     return (total_time / n) * 1000  # ms/图像
#
# def get_memory_usage(device):
#     if device.type == "cuda":
#         return T.cuda.max_memory_allocated(device=device) / 1024 ** 2
#     else:
#         return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
#
# if device.type == "cuda":
#     T.cuda.reset_peak_memory_stats(device=device)
# speed = test_speed(model, validation_dl, device)
# mem_usage = get_memory_usage(device)
# print(f"\n推理平均测试速度: {speed:.3f} ms/图像")
# print(f"推理最大显存占用: {mem_usage:.2f} MB")
#
# # =================== 12. 测试集预测与提交 =====================
# model.eval()
# preds = []
# ids = []
# for batch in tqdm(test_dl, desc="Test Inference"):
#     imgs = batch if not isinstance(batch, (list, tuple)) else batch[0]
#     imgs = imgs.to(device)
#     with T.no_grad():
#         output = model(imgs)[0]
#         probs = T.softmax(output, dim=1).cpu().numpy()
#         preds.extend(probs)
#     if isinstance(batch, (list, tuple)):
#         ids.extend(batch[1])
#     else:
#         # 此时test_dl必须返回(img, id)
#         raise ValueError("Test DataLoader返回格式不符，应为(img, id)")
#
# submit_path = os.path.join(SAVE_DIR, 'submission.csv')
# with open(submit_path, 'w', encoding='utf-8') as f:
#     f.write('id,' + ','.join(le.inverse_transform(list(range(len(le.classes_))))) + '\n')
#     for i, output in zip(test_data['id'], preds):
#         f.write(i + ',' + ','.join([str(n) for n in output]) + '\n')
# print(f"Submission file saved at: {submit_path}")





import os
import time
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch as T
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import psutil
from thop import profile
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# =================== 路径与设备 =====================
DATA_DIR = r'/kaggle/input/dog-breed-identification'
SAVE_DIR = r'/kaggle/working/'
os.makedirs(SAVE_DIR, exist_ok=True)
TRAIN_IMG_PATH = os.path.join(DATA_DIR, 'train')
TEST_IMG_PATH = os.path.join(DATA_DIR, 'test')
LABELS_PATH = os.path.join(DATA_DIR, 'labels.csv')
device = T.device('cuda' if T.cuda.is_available() else 'cpu')

# =================== 1. 数据读取 =====================
train_data = pd.read_csv(LABELS_PATH)
test_files = [f for f in os.listdir(TEST_IMG_PATH) if f.endswith('.jpg')]
test_data = pd.DataFrame({'id': [os.path.splitext(f)[0] for f in test_files]})

# 标签编码
le = LabelEncoder()
train_data['breed'] = le.fit_transform(train_data['breed'])

# =================== 2. 数据集类 =====================
# class Dog_Breed_Dataset(Dataset):
#     def __init__(self, df: pd.DataFrame, img_base_path: str, split: str, transforms=None):
#         self.df = df
#         self.img_base_path = img_base_path
#         self.split = split
#         self.transforms = transforms

#     # def __getitem__(self, index):
#     #     img_path = os.path.join(self.img_base_path, self.df.loc[index, 'id'] + '.jpg')
#     #     img = Image.open(img_path).convert('RGB')
#     #     if self.transforms:
#     #         img = self.transforms(img)
#     #     if self.split != 'test':
#     #         y = self.df.loc[index, 'breed']
#     #         return img, y
#     #     else:
#     #         return img

#     def __getitem__(self, index):
#     img_path = os.path.join(self.img_base_path, self.df.loc[index, 'id'] + '.jpg')
#     img = Image.open(img_path).convert('RGB')
#     if self.transforms:
#         img = self.transforms(img)
#     if self.split != 'test':
#         y = self.df.loc[index, 'breed']
#         return img, y
#     else:
#         img_id = self.df.loc[index, 'id']
#         return img, img_id   # 这里多返回id


#     def __len__(self):
#         return len(self.df)

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
            img_id = self.df.loc[index, 'id']
            return img, img_id

    def __len__(self):
        return len(self.df)

# =================== 3. 增强与数据加载 =====================
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
train, val = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['breed'])
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
train_dataset = Dog_Breed_Dataset(train, TRAIN_IMG_PATH, 'train', train_transforms)
validation_dataset = Dog_Breed_Dataset(val, TRAIN_IMG_PATH, 'val', test_transforms)
test_dataset = Dog_Breed_Dataset(test_data, TEST_IMG_PATH, 'test', test_transforms)
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
validation_dl = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# =================== 4. 投票法 =====================
def voter(pred1, pred2, pred3, pred4, pred5):
    final_predictions = []
    for i in range(pred1.size(0)):
        preds = [pred1[i].item(), pred2[i].item(), pred3[i].item(), pred4[i].item(), pred5[i].item()]
        counts = pd.Series(preds).value_counts()
        pred = counts.index[0]
        final_predictions.append(pred)
    return T.tensor(final_predictions)

# =================== 5. 集成模型定义 =====================
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
        self.fc = nn.Linear(2048 * 2, num_classes)
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

# =================== 6. 模型统计 =====================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def model_size_mb(model):
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb
def get_flops(model, input_size=(1, 3, 224, 224)):
    x = T.randn(input_size).to(device)
    macs, params = profile(model, inputs=(x,), verbose=False)
    flops = 2 * macs
    return flops, params

# =================== 7. 训练主循环 =====================
def train_model(train_dl, val_dl, model, epochs=50):
    total_train_loss, total_train_accs = [], []
    total_valid_loss, total_valid_accs = [], []
    train_top5s, valid_top5s = [], []
    best_val_loss = float('inf')
    best_top1, best_top5 = 0, 0
    best_epoch = 0
    weights = model.get_weights()
    start_train_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss, train_accs, train_labels, train_preds = [], [], [], []
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
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
            acc = (T.argmax(T.softmax(preds[0], dim=1), dim=1) == y).float().mean()
            train_loss.append(loss.item() * x.size(0))
            train_accs.append(acc.item() * x.size(0))
            train_labels.extend(y.cpu().numpy())
            train_preds.extend(preds[0].detach().cpu().numpy())
        n_train = len(train_dl.dataset)
        avg_train_loss = np.sum(train_loss) / n_train
        avg_train_acc = np.sum(train_accs) / n_train
        train_top5 = top_k_accuracy_score(train_labels, np.vstack(train_preds), k=5)
        total_train_loss.append(avg_train_loss)
        total_train_accs.append(avg_train_acc)
        train_top5s.append(train_top5)

        # 验证
        model.eval()
        valid_loss, valid_accs, valid_labels, valid_preds = [], [], [], []
        for x, y in tqdm(val_dl, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
            x, y = x.to(device), y.long().to(device)
            with T.no_grad():
                preds = model(x)
                loss = sum(model.criterion(p, y) for p in preds) / len(preds)
            acc = (T.argmax(T.softmax(preds[0], dim=1), dim=1) == y).float().mean()
            valid_loss.append(loss.item() * x.size(0))
            valid_accs.append(acc.item() * x.size(0))
            valid_labels.extend(y.cpu().numpy())
            valid_preds.extend(preds[0].detach().cpu().numpy())
        n_valid = len(val_dl.dataset)
        avg_valid_loss = np.sum(valid_loss) / n_valid
        avg_valid_acc = np.sum(valid_accs) / n_valid
        valid_top5 = top_k_accuracy_score(valid_labels, np.vstack(valid_preds), k=5)
        total_valid_loss.append(avg_valid_loss)
        total_valid_accs.append(avg_valid_acc)
        valid_top5s.append(valid_top5)

        print(f"[Train | {epoch+1:03d}/{epochs}] loss = {avg_train_loss:.5f}, acc = {avg_train_acc:.5f}, top5 = {train_top5:.5f}")
        print(f"[Valid | {epoch+1:03d}/{epochs}] loss = {avg_valid_loss:.5f}, acc = {avg_valid_acc:.5f}, top5 = {valid_top5:.5f}")

        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            best_top1 = avg_valid_acc
            best_top5 = valid_top5
            best_epoch = epoch + 1
            weights = model.get_weights()
            T.save(weights, os.path.join(SAVE_DIR, 'best_ensemble.pth'))
            print(f">>> Model saved at epoch {best_epoch}, val_loss = {best_val_loss:.5f}")

    total_train_time = time.time() - start_train_time
    model.load_weights(weights)
    print(f"\n训练收敛到最优Val Loss的Epoch: {best_epoch}, 总训练用时: {total_train_time/60:.2f} 分钟")
    print(f"最优Top-1: {best_top1:.4f}, 最优Top-5: {best_top5:.4f}")

    # 返回历史指标
    history = {
        'train_loss': total_train_loss,
        'val_loss': total_valid_loss,
        'val_acc': total_valid_accs,
        'val_top5': valid_top5s
    }
    return history, total_train_time

# =================== 8. 可视化 =====================
def plot_metrics(history, save_dir, suffix="ensemble"):
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(save_dir, f"loss_curve_{suffix}.png"))
    plt.close()

    plt.figure()
    plt.plot(history['val_acc'], label='val_acc')
    plt.plot(history['val_top5'], label='val_top5')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(os.path.join(save_dir, f"acc_curve_{suffix}.png"))
    plt.close()

# =================== 9. 训练与保存 =====================
model = Model(inception, resnet50, num_classes=len(le.classes_))
print("\n====== 模型参数统计 ======")
print(f"参数量: {count_parameters(model)/1e6:.2f} M")
print(f"模型大小: {model_size_mb(model):.2f} MB")
flops, _ = get_flops(model, input_size=(1, 3, 224, 224))
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs/图像\n")

history, train_time = train_model(train_dl, validation_dl, model, epochs=6)
plot_metrics(history, SAVE_DIR, suffix="ensemble")

# =================== 10. 验证集指标（最终） =====================
model.load_weights(torch.load(os.path.join(SAVE_DIR, 'best_ensemble.pth')))
model.eval()
y_true, y_pred, y_pred_logits = [], [], []
for batch in tqdm(validation_dl, desc="Final Valid Eval"):
    imgs, breeds = batch
    imgs = imgs.to(device)
    breeds = breeds.to(device)
    with T.no_grad():
        preds = model(imgs)
        out = preds[0]
        pred_labels = out.argmax(dim=1)
        y_true.extend(breeds.cpu().numpy())
        y_pred.extend(pred_labels.cpu().numpy())
        y_pred_logits.extend(out.cpu().numpy())
val_top1_acc = accuracy_score(y_true, y_pred)
val_top5_acc = top_k_accuracy_score(y_true, np.vstack(y_pred_logits), k=5)
print(f"\nFinal Validation Top-1 Acc: {val_top1_acc:.4f}, Top-5 Acc: {val_top5_acc:.4f}")

# =================== 11. 推理速度与内存占用 =====================
def test_speed(model, dataloader, device):
    model.eval()
    total_time = 0
    n = 0
    with T.no_grad():
        for batch in tqdm(dataloader, desc="SpeedTest"):
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            if device.type == "cuda":
                T.cuda.synchronize()
            start = time.time()
            _ = model(imgs)
            if device.type == "cuda":
                T.cuda.synchronize()
            total_time += (time.time() - start)
            n += batch_size
    return (total_time / n) * 1000  # ms/图像

def get_memory_usage(device):
    if device.type == "cuda":
        return T.cuda.max_memory_allocated(device=device) / 1024 ** 2
    else:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

if device.type == "cuda":
    T.cuda.reset_peak_memory_stats(device=device)
speed = test_speed(model, validation_dl, device)
mem_usage = get_memory_usage(device)
print(f"\n推理平均测试速度: {speed:.3f} ms/图像")
print(f"推理最大显存占用: {mem_usage:.2f} MB")

# =================== 12. 测试集预测与提交 =====================
# model.eval()
# preds = []
# ids = []
# for batch in tqdm(test_dl, desc="Test Inference"):
#     imgs = batch if not isinstance(batch, (list, tuple)) else batch[0]
#     imgs = imgs.to(device)
#     with T.no_grad():
#         output = model(imgs)[0]
#         probs = T.softmax(output, dim=1).cpu().numpy()
#         preds.extend(probs)
#     if isinstance(batch, (list, tuple)):
#         ids.extend(batch[1])
#     else:
#         # 此时test_dl必须返回(img, id)
#         raise ValueError("Test DataLoader返回格式不符，应为(img, id)")

model.eval()
preds = []
ids = []
for batch in tqdm(test_dl, desc="Test Inference"):
    imgs, batch_ids = batch   # unpack
    imgs = imgs.to(device)
    with T.no_grad():
        output = model(imgs)[0]
        probs = T.softmax(output, dim=1).cpu().numpy()
        preds.extend(probs)
    ids.extend(batch_ids)     # ids累计

submit_path = os.path.join(SAVE_DIR, 'submission.csv')
with open(submit_path, 'w', encoding='utf-8') as f:
    f.write('id,' + ','.join(le.inverse_transform(list(range(len(le.classes_))))) + '\n')
    for i, output in zip(test_data['id'], preds):
        f.write(i + ',' + ','.join([str(n) for n in output]) + '\n')
print(f"Submission file saved at: {submit_path}")
