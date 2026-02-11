import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Visualization

import seaborn as sns
import matplotlib.pyplot as plt

# PyTorch

import torch as T
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = T.device('cuda' if T.cuda.is_available() else 'cpu')

# Read csv file
train_data = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
# Train data shape
print(f"Train dataset shape: {train_data.shape}")
# Sample of the train_data DataFrame
train_data.head()
test_data = pd.DataFrame([])
for dirname, _, filename in os.walk('/kaggle/input/dog-breed-identification/test/'):
    filename = pd.Series(filename)
    test_data = pd.concat([test_data, filename], axis=0)
test_data.columns = ['id']
test_data['id'] = test_data['id'].str.replace(".jpg","")
# Dataset shape
print(f"Test dataset shape: {test_data.shape}")
# Sample of the train_data DataFrame
test_data.head()
le = LabelEncoder()
train_data.loc[:,'breed'] = le.fit_transform(train_data.loc[:,'breed'])
train_data.head()


class Dog_Breed_Dataset(Dataset):

    def __init__(self, df: pd.DataFrame, img_base_path: str, split: str, transforms=None):
        self.df = df
        self.img_base_path = img_base_path
        self.split = split
        self.transforms = transforms

    def __getitem__(self, index):
        # Path of the image
        img_path = os.path.join(self.img_base_path + self.df.loc[index, 'id'] + '.jpg')
        # Read the image
        img = Image.open(img_path)
        # Perform the transformations
        if self.transforms:
            img = self.transforms(img)
        if self.split != 'test':
            y = self.df.loc[index, 'breed']
            return img, y
        else:
            return img

    def __len__(self):
        return len(self.df)



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

train_dataset = Dog_Breed_Dataset(
    df=train,
    img_base_path='/kaggle/input/dog-breed-identification/train/',
    split='train',
    transforms=train_transforms
)
validation_dataset = Dog_Breed_Dataset(
    df=val,
    img_base_path='/kaggle/input/dog-breed-identification/train/',
    split='val',
    transforms=test_transforms
)
test_dataset = Dog_Breed_Dataset(
    df=test_data,
    img_base_path='/kaggle/input/dog-breed-identification/test/',
    split='test',
    transforms=test_transforms
)

train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
validation_dl = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=4)
test_dl = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

print(f"Train data length: {len(train_dl.dataset)}, Validation data length: {len(validation_dl.dataset)}, Test data length: {len(test_dl.dataset)}")


def train_model(train_dl, val_dl, model, epochs=6):
    # history record
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    # Best validation loss
    best_val_loss = 1_000_000.0
    # Get initial weights
    weights = model.get_weights()

    for epoch in range(epochs):
        print("=" * 20, "Epoch: ", str(epoch), "=" * 20)

        train_correct_pred = 0
        val_correct_pred = 0
        train_acc = 0
        val_acc = 0
        train_loss = 0
        val_loss = 0

        # Set to training mode
        model.train()

        for x, y in train_dl:
            # Convert data to Tensor
            x = x.clone().detach().to(device).requires_grad_(True)
            y = y.clone().detach().long().to(device)
            # Reset gradients
            model.optim.zero_grad()
            model.optim_resnet.zero_grad()
            model.optim_inception.zero_grad()
            # Predict
            preds = model(x)

            # Compute the loss
            loss1 = model.criterion(preds[0], y)
            loss2 = model.criterion(preds[1], y)
            loss3 = model.criterion(preds[2], y)
            loss4 = model.criterion(preds[3], y)
            loss5 = model.criterion(preds[4], y)

            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            # Compute the gradients
            loss.backward()
            # Update weights
            model.optim.step()
            model.optim_resnet.step()
            model.optim_inception.step()

            train_loss += loss.item()

            pred1 = T.argmax(T.nn.functional.softmax(preds[0], dim=1), dim=1)
            pred2 = T.argmax(T.nn.functional.softmax(preds[1], dim=1), dim=1)
            pred3 = T.argmax(T.nn.functional.softmax(preds[2], dim=1), dim=1)
            pred4 = T.argmax(T.nn.functional.softmax(preds[3], dim=1), dim=1)
            pred5 = T.argmax(T.nn.functional.softmax(preds[4], dim=1), dim=1)

            final_pred = voter(pred1, pred2, pred3, pred4, pred5)
            train_correct_pred += (final_pred.long().unsqueeze(1) == y.unsqueeze(1).cpu()).sum().item()

        train_acc = train_correct_pred / len(train_dl.dataset)

        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)

        # Switch to evaluation mode
        model.eval()

        with T.no_grad():
            for x, y in val_dl:
                # Convert data to Tensor
                x = x.clone().detach().to(device)
                y = y.clone().detach().long().to(device)
                # Predict
                preds = model(x)
                # Compute the loss
                loss1 = model.criterion(preds[0], y)
                loss2 = model.criterion(preds[1], y)
                loss3 = model.criterion(preds[2], y)
                loss4 = model.criterion(preds[3], y)
                loss5 = model.criterion(preds[4], y)

                loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5

                pred1 = T.argmax(T.nn.functional.softmax(preds[0], dim=1), dim=1)
                pred2 = T.argmax(T.nn.functional.softmax(preds[1], dim=1), dim=1)
                pred3 = T.argmax(T.nn.functional.softmax(preds[2], dim=1), dim=1)
                pred4 = T.argmax(T.nn.functional.softmax(preds[3], dim=1), dim=1)
                pred5 = T.argmax(T.nn.functional.softmax(preds[4], dim=1), dim=1)

                final_pred = voter(pred1, pred2, pred3, pred4, pred5)

                val_correct_pred += (final_pred.long().unsqueeze(1) == y.unsqueeze(1).cpu()).sum().item()

                val_loss += loss.item()

        model.scheduler.step(val_loss)

        val_acc = val_correct_pred / len(val_dl.dataset)

        val_acc_history.append(val_acc)

        val_loss_history.append(val_loss)

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            weights = model.get_weights()

        print("Train acc: {:.4f} | Train Loss: {:.4f} | Validation acc: {:.4f} | Validation Loss: {:.4f}".format(
            train_acc, train_loss, val_acc, val_loss))
    model.load_weights(weights)

    model.eval()
    # Predictions DataFrame
    prob_preds = pd.DataFrame([])
    with T.no_grad():
        for x in test_dl:
            # Convert data to Tensor
            x = x.clone().detach().to(device)
            # Predict
            preds = model(x)

            pred1 = T.nn.functional.softmax(preds[0], dim=1)
            pred2 = T.nn.functional.softmax(preds[1], dim=1)
            pred3 = T.nn.functional.softmax(preds[2], dim=1)
            pred4 = T.nn.functional.softmax(preds[3], dim=1)
            pred5 = T.nn.functional.softmax(preds[4], dim=1)

            prob_pred = (pred1 + pred2 + pred3 + pred4 + pred5) / 5

            prob_pred = prob_pred.detach().cpu().numpy()
            prob_pred = pd.DataFrame(prob_pred)

            prob_preds = pd.concat([prob_preds, prob_pred], axis=0)

    return [train_acc_history, train_loss_history, val_acc_history, val_loss_history], prob_preds


def voter(pred1, pred2, pred3, pred4, pred5):
    """
        Get the final prediction through Majority Voting.
    """

    final_predictions = []

    for i in range(pred1.size(0)):

        count_df = pd.DataFrame(columns=['classes', 'count'])

        prediction1 = pd.DataFrame({'classes': [pred1[i].item()], "count": [1]})
        count_df = pd.concat([count_df, prediction1], axis=0)

        if count_df['classes'].isin([pred2[i].item()]).sum() == 1:
            count_df.loc[count_df['classes'].isin([pred2[i].item()]), 'count'] += 1
        else:
            prediction2 = pd.DataFrame({'classes': [pred2[i].item()], "count": [1]})
            count_df = pd.concat([count_df, prediction2], axis=0)

        if count_df['classes'].isin([pred3[i].item()]).sum() == 1:
            count_df.loc[count_df['classes'].isin([pred3[i].item()]), 'count'] += 1
        else:
            prediction3 = pd.DataFrame({'classes': [pred3[i].item()], "count": [1]})
            count_df = pd.concat([count_df, prediction3], axis=0)

        if count_df['classes'].isin([pred4[i].item()]).sum() == 1:
            count_df.loc[count_df['classes'].isin([pred4[i].item()]), 'count'] += 1
        else:
            prediction4 = pd.DataFrame({'classes': [pred4[i].item()], "count": [1]})
            count_df = pd.concat([count_df, prediction4], axis=0)

        if count_df['classes'].isin([pred5[i].item()]).sum() == 1:
            count_df.loc[count_df['classes'].isin([pred5[i].item()]), 'count'] += 1
        else:
            prediction5 = pd.DataFrame({'classes': [pred5[i].item()], "count": [1]})
            count_df = pd.concat([count_df, prediction5], axis=0)

        if len(count_df.loc[count_df['count'] == count_df['count'].max(), 'classes'].values) > 1:
            pred = count_df.loc[count_df['count'] == count_df['count'].max(), 'classes'].values
            pred = int(pred[0])
        else:
            pred = int(count_df.loc[count_df['count'] == count_df['count'].max(), 'classes'].values)
        final_predictions.append(pred)
    return T.tensor(final_predictions)

inception = models.inception_v3(pretrained=True)

inception_model = nn.Sequential(
    inception.Conv2d_1a_3x3,
    inception.Conv2d_2a_3x3,
    inception.Conv2d_2b_3x3,
    inception.maxpool1,
    inception.Conv2d_3b_1x1,
    inception.Conv2d_4a_3x3,
    inception.maxpool2,
    inception.Mixed_5b,
    inception.Mixed_5c,
    inception.Mixed_5d,
    inception.Mixed_6a,
    inception.Mixed_6b,
    inception.Mixed_6c,
    inception.Mixed_6d,
    inception.Mixed_6e,
    inception.Mixed_7a,
    inception.Mixed_7b,
    inception.Mixed_7c,
    inception.avgpool
)

resnet50 = models.resnet50(pretrained=True)

resnet50_model = nn.Sequential(
    resnet50.conv1,
    resnet50.bn1,
    resnet50.relu,
    resnet50.maxpool,
    resnet50.layer1,
    resnet50.layer2,
    resnet50.layer3,
    resnet50.layer4,
    resnet50.avgpool
)


class Model(nn.Module):

    def __init__(self, inception_model, resnet50_model):
        super(Model, self).__init__()

        self.inception_model = inception_model
        self.resnet50_model = resnet50_model

        self.output = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(4096, 120)
        )

        self.to(device)
        # Optimizer
        self.optim = T.optim.SGD(self.output.parameters(), lr=0.005, momentum=0.9)
        self.optim_resnet = T.optim.Adam(self.resnet50_model.parameters(), lr=0.0001)
        self.optim_inception = T.optim.Adam(self.inception_model.parameters(), lr=0.0001)
        # Loss
        self.criterion = T.nn.CrossEntropyLoss()
        # Scheduler
        self.scheduler = ReduceLROnPlateau(self.optim, mode='min', factor=0.1, patience=5)

    def forward(self, x):
        X1 = self.inception_model(x)
        X2 = self.resnet50_model(x)

        X1 = X1.view(X1.size(0), -1)
        X2 = X2.view(X2.size(0), -1)

        X = T.cat([X1, X2], dim=1)

        P1 = self.output(X)
        P2 = self.output(X)
        P3 = self.output(X)
        P4 = self.output(X)
        P5 = self.output(X)

        return [P1, P2, P3, P4, P5]

    def get_weights(self):
        return self.state_dict()

    def load_weights(self, weights):
        self.load_state_dict(weights)

model = Model(inception_model, resnet50_model)

history, test_preds = train_model(train_dl, validation_dl, model)


# Training and Validation Results
fig, axs = plt.subplots(1,2, figsize=(15,6))
axs[0].plot(range(6), history[0], label="Training accuracy")
axs[0].plot(range(6), history[2], label="Validation accuracy")
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].grid(True)

axs[1].plot(range(6), history[1], label="Training Loss")
axs[1].plot(range(6), history[3], label="Validation Loss")
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].grid(True)

plt.suptitle("Training and Validation Results of Model")
plt.legend()
plt.show()

test_preds.head()

# Set columns to breed names
num_classes = []
for num_class in test_preds.columns:
    num_classes.append(num_class)

num_classes = np.array(num_classes)
num_classes = le.inverse_transform(num_classes)
test_preds.columns = list(num_classes)

test_preds.head()

# Set id column
test_preds = test_preds.reset_index(drop=True)
ids = test_data.loc[:,'id']
test_preds = pd.concat([ids, test_preds], axis=1)
test_preds.head()

test_preds.to_csv('submission.csv', index=None)