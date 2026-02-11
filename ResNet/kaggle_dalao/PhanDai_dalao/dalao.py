import pandas as pd
import numpy as np
import torch
#using GPU for faster training
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_cuda=torch.cuda.is_available()
# read the csv
labels = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
sample_submission = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")

labels.head()
sample_submission.head()
labels.info()
sample_submission.info()
labels["breed"].describe()
labels["breed"].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(30,20))
sns.countplot(y="breed",data=labels,palette="Set1") # using palette set1 for better visualization
plt.show()

import matplotlib.pyplot as plt
import random
from PIL import Image
import pandas as pd
df = labels

image_dir = "/kaggle/input/dog-breed-identification/train/"

df['image_path'] = df['id'].apply(lambda x: f"{image_dir}{x}.jpg")

random_samples = df.sample(25,replace=True)

fig, axes = plt.subplots(5, 5, figsize=(15, 15))
axes = axes.flatten()

for i, (index, row) in enumerate(random_samples.iterrows()):
    img = Image.open(row['image_path'])
    axes[i].imshow(img)
    axes[i].set_title(row['breed'], fontsize=12, loc='center')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

df["image_path"][0]

breed=list(df["breed"].value_counts().keys())
np.random.seed(42)
labels.groupby('breed').count().sort_values(by='id',ascending=False)


from torchvision import transforms
#Resizing images to fit Resnet18 architecture
train_trasforms=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Apply color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] # This is from the original paper
)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

labels['breed']=encoder.fit_transform(labels['breed'])

labels['breed']

# stratified train val split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(labels,labels['breed'],test_size=0.2,random_state=42,stratify=labels["breed"])

y_train

X_train=pd.DataFrame(X_train)
y_train=pd.DataFrame(y_train)

from torch.utils.data import Dataset, DataLoader


class DogBreedDataset(Dataset):
    def __init__(self, root_dir="/kaggle/input/dog-breed-identification/train", transforms=None, labels=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels[idx]['id']
        img_path = os.path.join(self.root_dir, img_name + '.jpg')
        image = Image.open(img_path)
        label = self.labels[idx]['breed']
        if self.transforms:
            image = self.transforms(image)
        return image, label


train_dataset=DogBreedDataset('/kaggle/input/dog-breed-identification/train',
                               transforms=train_trasforms,
                               labels=X_train.to_dict('records'))

train_dataloader=DataLoader(train_dataset, batch_size=32, shuffle=True)

val_trasforms=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
val_dataset=DogBreedDataset('/kaggle/input/dog-breed-identification/train',
                               transforms=val_trasforms,
                               labels=X_val.to_dict('records'))

val_dataloader=DataLoader(val_dataset, batch_size=32, shuffle=True)

import torchvision.models as models

net = models.resnet18(pretrained=True)

num_classes=len(labels['breed'].unique())

import torch.nn as nn
# change the last layer
net.fc = nn.Sequential(
    nn.Dropout(0.4),  # Dropout with a probability of 0.4
    nn.Linear(net.fc.in_features, 512),
    nn.BatchNorm1d(512),
    nn.LeakyReLU(negative_slope=0.01),
    nn.Linear(512,num_classes))
net = net.cuda() if use_cuda else net # move net to cuda


criterion=nn.CrossEntropyLoss()
learning_rate=0.0001
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate) # train the whole model
# ~100M params
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

import os

torch.manual_seed(42)  # fixing the seed

train_losses = []  # to save train losses
val_losses = []  # to save test losses

epochs = 20
for epoch in range(epochs):
    # Training
    net.train()  # in training mode
    train_loss = 0.0
    for features, target in train_dataloader:  # iterate over the train dataloader
        optimizer.zero_grad()  # to set gradient to zero at the begnining of each epoch
        features = features.to(device).double()
        target = target.to(device)
        outputs = net(features.float())  # calculating outputs from the features
        loss = criterion(outputs, target)  # calculate loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * features.size(0)
    train_loss /= len(train_dataloader.dataset)
    train_losses.append(train_loss)

    # Evaluation
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features_t, target_t in val_dataloader:
            features_t = features_t.to(device).double()
            target_t = target_t.to(device)
            outputs_t = net(features_t.float())
            loss_t = criterion(outputs_t, target_t)
            val_loss += loss_t.item() * features_t.size(0)  # Use features_t.size(0) for val_loss accumulation
            _, pred_t = torch.max(outputs_t, 1)
            total += target_t.size(0)  # Use target_t.size(0) to update total
            correct += (pred_t == target_t).sum().item()
    val_loss /= len(val_dataloader.dataset)
    val_losses.append(val_loss)
    scheduler.step()

    if epoch % 1 == 0:
        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {val_loss:.4f}, Test Accuracy: {(100 * correct / total):.2f}%')

import matplotlib.pyplot as plt

# Assuming you have trainlosses and val_losses lists
epochs = list(range(1, 21))  # 20 epochs
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
plt.plot(epochs, val_losses, label='Validation Loss', color='orange', marker='o')

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


# custom class to read and transform the test data
class Dog_Breed_Dataset_test(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # directory of images
        self.transforms = transform  # transformation
        self.image_names = [f.split('.')[0] for f in
                            os.listdir(root_dir)]  # image name is the first element after splitting on "."

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]  # read images names from csv
        img_path = os.path.join(self.root_dir, img_name + '.jpg')  # construct image path from the csv and directory
        image = Image.open(img_path)

        if self.transforms:  # if there is a transformation , apply it to data and then return image and its name
            image = self.transforms(image)
        return image, img_name

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

kaggle_test=Dog_Breed_Dataset_test(root_dir='/kaggle/input/dog-breed-identification/test',
                                  transform=test_transforms)

#kaggla test data loader
kaggle_test_dataloader=DataLoader(kaggle_test,batch_size=32,shuffle=False)

dog_breeds = encoder.classes_

# import torch
import torch.nn.functional as F

# Make sure the model is in evaluation mode
net.eval()

# List to store all predictions
all_predictions = []

# Disable gradient calculations
with torch.no_grad():
    for features_t, img_names in kaggle_test_dataloader:
        features_t = features_t.to(
            device).double()  # Move features to the appropriate device and ensure they are double type

        # Get model predictions
        outputs_t = net(features_t.float())  # Get the raw outputs from the model

        # Convert raw outputs to probabilities using softmax
        probabilities = F.softmax(outputs_t, dim=1)

        # Append probabilities to the list of all predictions
        # Extract probabilities for each class for each image
        for i in range(len(img_names)):
            probabilities_for_image = probabilities[i].cpu().numpy()
            for class_idx, prob in enumerate(probabilities_for_image):
                class_name = dog_breeds[class_idx]  # Assuming class_names is a list containing the class names
                all_predictions.append({'class_name': class_name, 'probability': prob, 'image_name': img_names[i]})


all_predictions

#preparing submissions file
all_predictions_df = pd.DataFrame.from_records(all_predictions)
submissions = all_predictions_df.pivot(index='image_name', columns='class_name', values='probability')

# Reset index to make 'image_name' a regular column instead of index
submissions.reset_index(inplace=True)

#writing the csv file for predictions
submissions=submissions.rename(columns={'image_name':'id'})
submissions.to_csv('submission.csv',index=False)