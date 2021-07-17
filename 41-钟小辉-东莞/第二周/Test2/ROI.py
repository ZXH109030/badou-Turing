import  numpy as np
import  matplotlib.pyplot as plt
import  cv2

def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img3 = cv2.imread("lenna.png")
roi = img3[0:200,0:200]
# cv2.imshow("image",roi)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()

#图像拆分
b,g,r = cv2.split(img3)
print(b)
print(b.shape)
#图像合并
img =cv2.merge((b,g,r))
print(img.shape)

#只保留b,g,r
cur_img = img.copy()
cur_img[:,:,0] = 0
cur_img[:,:,1] = 0
cv_show("r",cur_img)
