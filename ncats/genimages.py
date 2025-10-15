import numpy as np
import matplotlib.pyplot as plt

# Línea 5
train_data_flattened = np.loadtxt('ncats\\dataset\\train\\train_images.csv', delimiter=',')  # (209,12288)
train_labels = np.loadtxt('ncats\\dataset\\train\\train_labels.csv', delimiter=',')  # (209,)

# Línea 8
test_data_flattened = np.loadtxt('ncats\\dataset\\test\\test_images.csv', delimiter=',')  # (50,12288)
test_labels = np.loadtxt('ncats\\dataset\\test\\test_labels.csv', delimiter=',')  # (50,)

# Normalize the data
train_data = train_data_flattened / 255.0
test_data = test_data_flattened / 255.0

train_data_img = train_data.reshape(209, 64, 64, 3)

for index in range(0, 209):
  plt.imsave("ncats/dataset/train/train_cat_"+ (str(index+1)) + ".png", train_data_img[index])

test_data_img = test_data.reshape(50, 64, 64, 3)

for index in range(0, 50):
  plt.imsave("ncats/dataset/train/test_cat_"+ (str(index+1)) + ".png", test_data_img[index])


