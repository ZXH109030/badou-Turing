import numpy as np
import matplotlib.pyplot as plt
import cv2
# %matplotlib inline
# cv2.IMREAD_GRAYSCALE #灰度图
# cv2.IMREAD_COLOR #彩色图

vc = cv2.VideoCapture("test.mp4")
# if vc.isOpened():
#     open,frame = vc.read()
# else:
#     open =False

while open:
    ret,frame =vc.read()
    if frame is None:
        break
    if ret == True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("result",gray)
        if cv2.waitKey(10)&0xFF ==27:
            break
vc.release()
cv2.destroyAllWindows()
