import numpy as np
import matplotlib.pyplot as plt
import cv2
# %matplotlib inline
# cv2.IMREAD_GRAYSCALE #灰度图
# cv2.IMREAD_COLOR #彩色图

def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("lenna.PNG")
# print(img)
# print(img.shape)
# cv_show("image",img)
print(type(img))

img2 = cv2.imread("lenna.png",cv2.IMREAD_GRAYSCALE)
cv_show("image",img2)

cv2.imwrite("1.png",img)


print(img2.size)
print(img2.dtype)

