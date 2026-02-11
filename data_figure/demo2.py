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
