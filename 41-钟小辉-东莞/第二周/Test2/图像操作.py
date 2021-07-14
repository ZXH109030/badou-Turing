import cv2
import matplotlib.pyplot as plt
import matplotlib_inline.config
import numpy as np
# %matplotlib_inline

#预处理
def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img =cv2.imread("C:/Users/ZhongXH2/Desktop/zuoye/cat.jpg")
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

###HSV(H色调，s饱和度，V强度 / RGB
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
h,s,v= cv2.split(img)
#这里可以对hsv进行矩阵操作
print(type(h))
print(h.dtype)
# cv_show("HSV",s)

h_img = img.copy()
h_img[:,:,0] = 0
h_img[:,:,1] = 0
# cv_show("HSV",h_img)

#二值化-只能操作灰度图 ret,dst = cv2.threshold(src, thresh, maxval, type)
# src： 输入图，只能输入单通道图像，通常来说为灰度图
# dst： 输出图
# thresh： 阈值
# maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
# type：二值化操作的类型，包含以下5种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV
# cv2.THRESH_BINARY
# 超过阈值部分取maxval（最大值），否则取0
# cv2.THRESH_BINARY_INV
# THRESH_BINARY的反转
# cv2.THRESH_TRUNC
# 大于阈值部分设为阈值，否则不变
# cv2.THRESH_TOZERO
# 大于阈值部分不改变，否则设为0
# cv2.THRESH_TOZERO_INV
# THRESH_TOZERO的反转

ret,thresh1 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img_gray,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO_INV)
titles = ['Original image',"THRESH_BINARY","THRESH_BINARY_INV","THRESH_TRUNC","THRESH_TOZERO","THRESH_TOZERO_INV"]
images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]

for ii in range(6):
    plt.subplot(2,3,ii+1),plt.imshow(images[ii],"gray")
    plt.title(titles[ii])
    plt.xticks([])
    plt.yticks([])
plt.show()

img = cv2.imread("C:/Users/ZhongXH2/Desktop/zuoye/lenaNoise.png")
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv_show("image",img)
# print(img[0:7,0:7])
# print("------------")

#均值滤波
blur = cv2.blur(img,(3,3))
# print(blur[0:7,0:7])
# cv_show("image",blur)

#方框滤波
box =cv2.boxFilter(img,-1,(3,3))

# cv_show("image",box)
#高斯滤波
aussian = cv2.GaussianBlur(img,(3,3),1)
# cv_show("aussian",aussian)

#中值滤波
median = cv2.medianBlur(img,3)
# cv_show("image",median)
#展示
res = np.hstack((blur,aussian,median))
# print(res)
cv_show("image",res)