import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import svm


def show_image(title,image):
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()


train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt', 'int32')

test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt', 'int32')

clf = svm.SVC()
clf.fit(train_images, train_labels)

print(clf.score(test_images, test_labels))

for i in range(1, 10):
    image = train_images[i, :]
    image = np.reshape(image, (28, 28))
    # plt.imshow(image.astype(np.uint8), cmap='gray')
    # plt.show()
    show_image("img", image)
    # # image = np.reshape(image, (35, 35))
# # print(500//9)