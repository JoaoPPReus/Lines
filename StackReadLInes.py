import cv2
import numpy as np

park = cv2.imread("Imagens/shutterstock-2.png")
gray = cv2.cvtColor(park, cv2.COLOR_BGR2GRAY)

imgBlur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Blur", imgBlur)

classifier = cv2.CascadeClassifier("Classificador\cascade3.xml")
carsDetected = classifier.detectMultiScale(imgBlur, 1.01, 2, minSize=(30,30))

imgCanny = cv2.Canny(imgBlur, 130, 130)
cv2.imshow("Cany", imgCanny)

lines = cv2.HoughLinesP(imgCanny, 0.1, (np.pi / 180), 5, np.array([]), 70, 70)

parkLine = np.copy(park) * 0
print(lines)

for (x, y, l, a) in carsDetected:
    cv2.circle(parkLine,((x + int(l / 2)),(y + int(a / 2))), 5, (0, 0, 255), 20)
    cv2.rectangle(parkLine, (x, y), (x + l, y + a), (0, 0, 255), 2)

#for line in lines:
#    for x, y, h, w in line:
#            #if (x == h and (y - w) > 110) or (y == w and (h - x) > 600):
#            if (x == h and 480 < (y - w) < 520):
#               cv2.line(parkLine, (x, y), (h, w), (0, 255, 0), 2)

cannyLines = cv2.addWeighted(park, 0.8, parkLine, 1, 0)
cv2.imshow("Hough", cannyLines)
cv2.waitKey(0)