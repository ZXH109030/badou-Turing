import  numpy as np
import  matplotlib.pyplot as plt
import  cv2

def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("lenna.png")
roi = img[0:200,0:200]
# # print(roi)
# cv_show("r",roi)
#图像拆分
# b,g,r = cv2.split(img3)
# print(b)
# print(b.shape)
# #图像合并
# img =cv2.merge((b,g,r))
# print(img.shape)
#
# #只保留b,g,r
# cur_img = img.copy()
# cur_img[:,:,0] = 0
# cur_img[:,:,1] = 0
# cv_show("r",cur_img)



#边界填充
top_size,bottom_size,left_size,right_size = (50,50,50,50)


replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=0)

import matplotlib.pyplot as plt
plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')

plt.show()

# #图像融合
img_cat = cv2.imread("C:/Users/ZhongXH2/Desktop/zuoye/cat.jpg")
img_dog = cv2.imread("C:/Users/ZhongXH2/Desktop/zuoye/dog.jpg")
# cv_show("image",img_dog)
#
# img_cat2 = img_cat+20
# print(img_cat2[:5,:10,0])
#
# #相当于% 256
# print((img_cat + img_cat2)[:5,:,0])
# print(cv2.add(img_cat,img_cat2)[:5,:,0])

#图像融合
print(img_cat.shape)
img_dog =cv2.resize(img_dog,(500,414))
print(img_dog.shape)
res = cv2.addWeighted(img_dog,0.4,img_cat,0.6,0)

res2 = cv2.resize(img,(0,0),fx=4,fy=4)
plt.imshow(res2)

plt.show()

# cv_show("result",res)