import numpy as np
import idx2numpy
from logistic_regression import LogisticRegression
import cv2 as cv

train_images = idx2numpy.convert_from_file("Dataset/Train/train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("Dataset/Train/train-labels.idx1-ubyte")
train_set = (train_images, train_labels)

test_images = idx2numpy.convert_from_file("Dataset/Test/t10k-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("Dataset/Test/t10k-labels.idx1-ubyte")
test_set = (test_images, test_labels)

def filter_data(data, criterion):
    images, labels = data

    images = images[labels == criterion] 
    images = images.reshape(-1, 784)/255
    labels = labels[labels == criterion]

    return (images, labels)

def merge_data(data_set: list):
    images, labels = data_set[0]

    for ds in data_set[1:]:
        i, l = ds
        images = np.concatenate([images, i], axis=0)
        labels = np.concatenate([labels, l], axis=0)

    return (images, labels)

train_set_0 = filter_data(train_set, 0)
train_set_1 = filter_data(train_set, 1)
train_set = merge_data([
    train_set_0,
    train_set_1
])

test_set_0 = filter_data(test_set, 0)
test_set_1 = filter_data(test_set, 1)
test_set = merge_data([
    test_set_0,
    test_set_1
])

train_images, train_labels = train_set
model = LogisticRegression(learning_rate=0.2, epochs=100)
model.fit(train_images, train_labels)

test_images, test_labels = test_set
metrics = model.evaluate(test_images, test_labels)
for key, value in metrics.items():
    print(f"{key}: {value}")

np.save("visualize/loss_history_logistic.npy", model.loss_history)
