import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models
import os
import shutil
import sys

dataset_dir = '../input/dog-breed-identification/train'
labels = pd.read_csv('../input/dog-breed-identification/labels.csv')

n_class = len(labels.breed.unique())
n_class

import os


def make_dir(x):
    if os.path.exists(x) == False:
        os.makedirs(x)


base_dir = './subset'
make_dir(base_dir)

train_dir = os.path.join(base_dir, 'train')
make_dir(train_dir)

breeds = labels.breed.unique()
for breed in breeds:
    # Make folder for each breed
    _ = os.path.join(train_dir, breed)
    make_dir(_)

    # Copy images to the corresponding folders
    images = labels[labels.breed == breed]['id']
    for image in images:
        source = os.path.join(dataset_dir, f'{image}.jpg')
        destination = os.path.join(train_dir, breed, f'{image}.jpg')
        shutil.copyfile(source, destination)

breeds

batch_size = 128

datagen = ImageDataGenerator(rescale=1./255, # rescale pixel values to [0,1] to reduce memory usage
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split


train_generator = datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training')

validation_generator = datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation')

from tensorflow.keras.applications import InceptionResNetV2

inception_bottleneck = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

from tensorflow.keras.applications.vgg19 import VGG19

vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

from tensorflow.keras.applications import InceptionV3

InceptionV3_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


InceptionV3_model.trainable = False

model = models.Sequential()
model.add(InceptionV3_model)
model.add(GlobalAveragePooling2D())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(n_class, activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.summary()

#early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)


history = model.fit(train_generator,
          #batch_size=batch_size,
          steps_per_epoch = train_generator.samples // batch_size,
          validation_data = validation_generator,
          validation_steps = validation_generator.samples // batch_size,
          epochs=5,
          verbose=1)
          #callbacks=[early_stop])

!mkdir test_data
!cp -r "/kaggle/input/dog-breed-identification/test" test_data

datagen_test = ImageDataGenerator(rescale=1./255) # rescale pixel values to [0,1] to reduce memory usage

test_generator = datagen_test.flow_from_directory(
    directory="test_data",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

images_names = os.listdir('../input/dog-breed-identification/test/')
images_ids = [id.split('.')[0] for id in images_names]

probabilities = model.predict(test_generator)

rows = []
for index, pred in enumerate(probabilities):
    pred_format = [format(z, '.4f') for z in pred]
    pred_format.insert(0, images_ids[index])
    rows.append(pred_format)

fields = breeds.tolist()
fields.insert(0, "id")

import csv

with open('submission.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(rows)