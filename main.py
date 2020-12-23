from scipy import misc

import matplotlib.pyplot as plt
import numpy as np
import math

import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

img = misc.imread('pentagon.png')

print('image shape: ', img.shape)

plt.imshow(img, )

plt.savefig("image.png",bbox_inches='tight')

plt.close()