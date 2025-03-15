import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import (binary_closing,binary_opening,binary_dilation, binary_erosion)
data = np.load("wires6npy.txt")

labeled = label(data)


result = binary_erosion(data,np.ones(3).reshape(3,1))


for i in range (1,np.max(labeled)+1):
    result1 = label(binary_erosion(labeled==i,np.ones(3).reshape(3,1)))
    res = np.max(result1) - 1
    if res > 0:
        print(f"The wire {i} is cut into {res} pieces")
    elif res == 0 :
        print(f"The wire {i} has no cuts")
    else:
        print(f"The wire {i} does not exist")

plt.subplot(121)
plt.imshow(data)
plt.subplot(122)
plt.imshow(result)
plt.show()