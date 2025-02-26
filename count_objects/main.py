import numpy as np
import matplotlib.pyplot as plt


external = np.diag([1, 1, 1, 1]).reshape(4, 2, 2)

internal = np.logical_not(external)

cross = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])


def match(a, masks):
    for mask in masks:
        if np.all((a!=0) == (mask!=0)):
            return True
    return False


def count_objects(image):
    E = 0
    for y in range(0, image.shape[0] - 1):
        for x in range(0, image.shape[1] - 1):
            sub = image[y : y + 2, x : x + 2]
            if match(sub, external):
                E += 1
            elif match(sub, internal):
                E -= 1
            elif match(sub, cross):
                E += 2
    return E / 4


image1 = np.load("example1.npy")
image2 = np.load("example2.npy")

print(image1.shape)
print(image2.shape)

plt.figure( figsize=(10,10))
plt.subplot(1,4,1)
plt.title("image1")
plt.imshow(image1)
plt.subplot(1,4,2)
plt.title("image2, 1st layer")
plt.imshow(image2[:,:,0])
plt.subplot(1,4,3)
plt.title("image2, 2nd layer")
plt.imshow(image2[:,:,1])
plt.subplot(1,4,4)
plt.title("image2, 3rd layer")
plt.imshow(image2[:,:,2])

print("Image1 figure count:  ",count_objects(image1))
arr_img2 = [count_objects(image2[:,:,0]),count_objects(image2[:,:,1]),count_objects(image2[:,:,2])]
print("Image2 figure count:  ",sum(arr_img2))

plt.show()

