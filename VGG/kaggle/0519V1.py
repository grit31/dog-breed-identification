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
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

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

# === Dataset ===
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

# === 数据增强选择 ===
def get_train_transform(aug_mode):
    if aug_mode == 'aug':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    elif aug_mode == 'none':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        raise ValueError("aug_mode must be 'none' or 'aug'.")

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def plot_metric(metric_list, metric_name, save_path):
    plt.figure()
    plt.plot(metric_list, marker='o')
    plt.title(f'{metric_name} curve')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {metric_name} curve to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fff', help='a dummy argument for Jupyter notebook', default=None)
    parser.add_argument('--train_dir', type=str, default='/kaggle/input/dog-breed-identification/train')
    parser.add_argument('--test_dir', type=str, default='/kaggle/input/dog-breed-identification/test')
    parser.add_argument('--labels', type=str, default='/kaggle/input/dog-breed-identification/labels.csv')
    parser.add_argument('--sample_submission', type=str, default='/kaggle/input/dog-breed-identification/sample_submission.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_csv', type=str, default='/kaggle/working/submission.csv')
    parser.add_argument('--aug_mode', type=str, default='none', choices=['none', 'aug'],
                        help="数据增强模式：none=无增强, aug=用增强")
    #parser.add_argument('--do_train', action='store_true', help='train model')
    #parser.add_argument('--do_test', action='store_true', help='test model')
    parser.add_argument('--do_train', default='True', help='train model')
    parser.add_argument('--do_test', default='False', help='test model')
    parser.add_argument('--model_path', type=str, default='/kaggle/working/vgg11best.pth')
    args = parser.parse_args()
    save_dir = '/kaggle/working/'
    print(f'Using device: {args.device}, data augmentation: {args.aug_mode}')

    # 1. 读取标签
    label_df = pd.read_csv(args.labels)
    breeds = sorted(label_df['breed'].unique())
    breed2idx = {breed: i for i, breed in enumerate(breeds)}
    idx2breed = {i: breed for breed, i in breed2idx.items()}
    num_classes = len(breeds)
    print(f'Classes: {num_classes}')

    # 2. 数据增强
    train_transform = get_train_transform(args.aug_mode)
    test_transform = get_test_transform()

    # 3. 数据集
    train_dataset = DogDataset(label_df, args.train_dir, breed2idx, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 4. 构建模型
    model = get_vgg11(num_classes)
    model = model.to(args.device)

    if args.do_train:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # 记录每轮的loss/acc等
        loss_history = []
        acc_history = []
        top3_history = []
        top5_history = []
        f1_history = []
        best_loss = float('inf')
        best_state = None

        # for epoch in range(args.epochs):
        #     model.train()
        #     running_loss = 0.0
        #     preds = []
        #     targets = []
        #     for images, labels in tqdm(train_loader):
        #         images = images.to(args.device)
        #         labels = labels.to(args.device)
        #         optimizer.zero_grad()
        #         outputs = model(images)
        #         loss = criterion(outputs, labels)
        #         loss.backward()
        #         optimizer.step()
        #         running_loss += loss.item() * images.size(0)
        #
        #         # For metrics
        #         pred_labels = outputs.argmax(dim=1).cpu().numpy()
        #         preds.append(pred_labels)
        #         targets.append(labels.cpu().numpy())
        #
        #     epoch_loss = running_loss / len(train_loader.dataset)
        #     preds = np.concatenate(preds)
        #     targets = np.concatenate(targets)
        #     epoch_acc = accuracy_score(targets, preds)
        #     # Top-3, Top-5准确率
        #     all_outputs = []
        #     for images, labels in DataLoader(train_dataset, batch_size=args.batch_size):
        #         images = images.to(args.device)
        #         with torch.no_grad():
        #             out = model(images)
        #         all_outputs.append(out.cpu().numpy())
        #     all_outputs = np.vstack(all_outputs)
        #     y_true = np.array([breed2idx[row['breed']] for _, row in train_dataset.df.iterrows()])
        #     # sklearn 1.0+: top_k_accuracy_score
        #     top3 = top_k_accuracy_score(y_true, all_outputs, k=3, labels=np.arange(num_classes))
        #     top5 = top_k_accuracy_score(y_true, all_outputs, k=5, labels=np.arange(num_classes))
        #     epoch_f1 = f1_score(targets, preds, average='macro')

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            preds = []
            targets = []
            probs_epoch = []
            for images, labels in tqdm(train_loader):
                images = images.to(args.device)
                labels = labels.to(args.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

                # 累计softmax概率与标签
                probs_batch = torch.softmax(outputs.detach(), dim=1).cpu().numpy()
                probs_epoch.append(probs_batch)
                targets.append(labels.cpu().numpy())

            # 拼接所有batch
            probs_epoch = np.concatenate(probs_epoch)
            targets = np.concatenate(targets)
            preds = np.argmax(probs_epoch, axis=1)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = accuracy_score(targets, preds)
            top3 = top_k_accuracy_score(targets, probs_epoch, k=3, labels=np.arange(num_classes))
            top5 = top_k_accuracy_score(targets, probs_epoch, k=5, labels=np.arange(num_classes))
            epoch_f1 = f1_score(targets, preds, average='macro')
            print(
                f"Epoch {epoch + 1}/{args.epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Top3: {top3:.4f} Top5: {top5:.4f} F1: {epoch_f1:.4f}")
            # ...（后续保存指标等和原来一样即可）

            print(f"Epoch {epoch+1}/{args.epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Top3: {top3:.4f} Top5: {top5:.4f} F1: {epoch_f1:.4f}")
            # 保存历史
            loss_history.append(epoch_loss)
            acc_history.append(epoch_acc)
            top3_history.append(top3)
            top5_history.append(top5)
            f1_history.append(epoch_f1)

            # current=model.state_dict()
            # torch.save(current, args.model_path)
            # print(f"Saved current model with loss {epoch_loss:.4f}")
            # 保存最佳
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = model.state_dict()
                torch.save(best_state, args.model_path)
                print(f"Saved best model with loss {best_loss:.4f}")

        # 曲线和数据保存
        pd.DataFrame({
            'loss': loss_history,
            'acc': acc_history,
            'top3': top3_history,
            'top5': top5_history,
            'f1': f1_history
        }).to_csv(os.path.join(save_dir, 'train_metric.csv'), index=False)
        plot_metric(loss_history, 'Loss',  os.path.join(save_dir, 'loss_curve.png'))
        plot_metric(acc_history, 'Accuracy', os.path.join(save_dir, 'acc_curve.png'))
        plot_metric(top3_history, 'Top3_Accuracy',  os.path.join(save_dir, 'top3_curve.png'))
        plot_metric(top5_history, 'Top5_Accuracy', os.path.join(save_dir, 'top5_curve.png'))
        plot_metric(f1_history, 'F1', os.path.join(save_dir, 'f1_curve.png'))
        print("Training curves and metrics saved.")

    if args.do_test:
        # 推理与生成提交
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

        # # 如有真实label，可在此加载并评估
        # # 否则只输出提示
        # print("Test集无标签，无法评估测试指标。如果你后续有测试集标签，只需加代码调用sklearn即可。")

if __name__ == "__main__":
    main()

