# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 读取标签文件
# labels = pd.read_csv(r'D:\homework\Course_Design_of_artificial_intelligence\third\dog-breed-identification\labels.csv')  # 路径改为实际数据集路径
#
# plt.figure(figsize=(14, 8))
# sns.countplot(data=labels, y='breed', order=labels['breed'].value_counts().index)
# plt.title('Samples per Class (Dog Breed)')
# plt.xlabel('Number of Samples')
# plt.ylabel('Breed')
# plt.tight_layout()
# plt.savefig('breed_sample_distribution.png')
# plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

labels = pd.read_csv(r'D:\homework\Course_Design_of_artificial_intelligence\third\dog-breed-identification\labels.csv')  # 路径按实际修改

plt.figure(figsize=(14, 24))  # 原为(14, 8)，将高度从8增加到24
sns.countplot(data=labels, y='breed', order=labels['breed'].value_counts().index)
plt.title('Samples per Class (Dog Breed)')
plt.xlabel('Number of Samples')
plt.ylabel('Breed')
plt.tight_layout()
plt.savefig('breed_sample_distribution_tall.png')
plt.show()

import os
from PIL import Image

# 图片文件夹路径
img_dir = r'D:\homework\Course_Design_of_artificial_intelligence\third\dog-breed-identification\train'  # 改为你的图片文件夹路径

widths, heights = [], []

for fname in labels['id'].values:
    img_path = os.path.join(img_dir, fname + '.jpg')
    try:
        img = Image.open(img_path)
        w, h = img.size
        widths.append(w)
        heights.append(h)
    except Exception as e:
        print(f"Warning: could not open {img_path}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(widths, bins=30, color='skyblue')
plt.xlabel('Width (pixels)')
plt.ylabel('Count')
plt.title('Image Width Distribution')

plt.subplot(1, 2, 2)
plt.hist(heights, bins=30, color='salmon')
plt.xlabel('Height (pixels)')
plt.ylabel('Count')
plt.title('Image Height Distribution')
plt.tight_layout()
plt.savefig('image_size_distribution.png')
plt.show()

import numpy as np

plt.figure(figsize=(14, 10))
breeds = labels['breed'].unique()
for i, breed in enumerate(np.random.choice(breeds, 8, replace=False)):  # 随机展示8类
    img_id = labels[labels['breed'] == breed].iloc[0]['id']
    img_path = os.path.join(img_dir, img_id + '.jpg')
    plt.subplot(2, 4, i+1)
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(breed)
    plt.axis('off')
plt.suptitle('Example Images for 8 Random Breeds')
plt.tight_layout()
plt.savefig('example_breeds.png')
plt.show()
