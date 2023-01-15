import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def drawImage(sample):
    img = sample.reshape((28,28))
    plt.imshow(img)
    plt.show()

def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X_images,Y_labels,testImage,k=5):
    print ("Start KNN")
    vals = []
    m = X_images.shape[0]
    for i in range(m):
        d = dist(testImage, X_images[i])
        vals.append((d,Y_labels[i]))
    
    print ("Sorting...")
    vals = sorted(vals)

    #Nearest/First K points
    vals = vals[:k]
    vals = np.array(vals)
    
    new_vals = np.unique(vals[:,1],return_counts = True)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred


ds = pd.read_csv('train.csv')
data = ds.values
# X = (data[:, 1:])
X_train = data[:,1:]
Y_train = data[:, 0]
# drawImage(X_train[29])
# print (X_train[0])


import cv2
test_image = cv2.imread('number.png', cv2.IMREAD_GRAYSCALE)
# # drawImage(test_image)
test_image_resized = cv2.resize(test_image, (28, 28), interpolation=cv2.INTER_LINEAR)
test_image_resized = cv2.bitwise_not(test_image_resized)
test_image_resized = test_image_resized.flatten()
# for idx, pix in enumerate(test_image_resized):
#     if pix < 130:
#         test_image_resized[idx] = 0
    # else:
        # test_image_resized[idx] = pix

# drawImage(test_image_resized)
# print (test_image_resized)
drawImage(test_image_resized)
# plt.show()
# print (test_image_resized)
print (int(knn(X_train, Y_train, test_image_resized)))