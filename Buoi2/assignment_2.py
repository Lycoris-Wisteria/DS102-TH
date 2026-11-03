import numpy as np
import idx2numpy
from softmax_regression import SoftmaxRegression
import cv2 as cv

def convert_to_one_hot(y: np.ndarray):
    K = np.max(y) + 1
    N = y.shape[0]
    oh_y = np.zeros((N, K))
    oh_y[np.arange(N), y] = 1

    return oh_y

if __name__ == "__main__":

    train_images = idx2numpy.convert_from_file("Dataset/Train/train-images.idx3-ubyte")
    train_labels = idx2numpy.convert_from_file("Dataset/Train/train-labels.idx1-ubyte")
    
    train_images= train_images.reshape(-1, 784)/255 # ổn định khoảng giá trị cho X đầu vào
    train_labels = convert_to_one_hot(train_labels)
    train_set=(train_images, train_labels)

    test_images = idx2numpy.convert_from_file("Dataset/Test/t10k-images.idx3-ubyte")
    test_labels = idx2numpy.convert_from_file("Dataset/Test/t10k-labels.idx1-ubyte")
    
    test_images= test_images.reshape(-1, 784)/255
    test_labels = convert_to_one_hot(test_labels)
    test_set = (test_images, test_labels)

    train_images, train_labels = train_set
    model = SoftmaxRegression(epochs=100, learning_rate=0.2)
    model.fit(train_images, train_labels)

    test_images, test_labels = test_set
    metrics = model.evaluate(test_images, test_labels)
    for key, value in metrics.items():
        print(f"{key}: {value}" )

np.save("visualize/loss_history_softmax.npy", model.loss_history)

