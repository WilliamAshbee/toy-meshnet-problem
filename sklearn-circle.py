from sklearn.feature_extraction import image
import numpy as np
l = 100
x, y = np.indices((l, l))

center1 = (28, 24)
radius1 = 16

circle1 = (x - center1[0])**2 + (y - center1[1])**2 < radius1**2

# 4 circles
img = circle1 
mask = img.astype(bool)
img = img.astype(float)

img += 1 + 0.2*np.random.randn(*img.shape)

print( type(img))

print(img.shape)
import png
img = (img>np.median(img)).astype(int)
img = img *255
w, h = img.shape
ret = np.empty((w, h, 3), dtype=np.uint8)
ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  img
from PIL import Image
i=Image.fromarray(ret,"RGB")
i.save("test.png")