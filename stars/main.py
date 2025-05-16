import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import (binary_closing,binary_opening,binary_dilation, binary_erosion)


data = np.load("./stars/stars.npy")

mask1 = np.array([[1,0,0,0,1],
                 [0,1,0,1,0],
                 [0,0,1,0,0],
                 [0,1,0,1,0],
                 [1,0,0,0,1]])

mask2 = np.array([[0,0,1,0,0],
                 [0,0,1,0,0],
                 [1,1,1,1,1],
                 [0,0,1,0,0],
                 [0,0,1,0,0]])



def match(sub, mask):
    if np.all(sub == mask):
        return  True
    return False

def count_objects(image):
    e = 0
    sizey = image.shape[0]
    sizex = image.shape[1]

    for y in range(0,sizey-4):
        for x in range(0,sizex-4):
            sub = image[y:y+5,x:x+5]
            if match(sub,mask1) or match(sub,mask2):
                e+=1
    return e
print(count_objects(data))

plt.imshow(data)

plt.show()
