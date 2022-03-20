import numpy as np
import cv2

img = cv2.imread('Imagens/shutterstock-2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(gray, 16, 50, 50)

corners = cv2.goodFeaturesToTrack(blur, 40, 0.1, 10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

cv2.imshow("Hough", img)
cv2.waitKey(0)