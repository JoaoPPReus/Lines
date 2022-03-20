import cv2
import numpy as np

video = cv2.VideoCapture("Video/video_estacionamento.mp4")

while True:
    conected, park = video.read()
    gray = cv2.cvtColor(park, cv2.COLOR_BGR2GRAY)

    imgBlur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Blur", imgBlur)

    classifier = cv2.CascadeClassifier("Classificador\Khare_classifier_01.xml")
    carsDetected = classifier.detectMultiScale(imgBlur, 1.02, 10)

    imgCanny = cv2.Canny(imgBlur, 130, 130)
    cv2.imshow("Cany", imgCanny)

    lines = cv2.HoughLinesP(imgCanny, 1, (np.pi / 180), 15, np.array([]), 50, 50)

    parkLine = np.copy(park) * 0
    print(lines)

    for (x, y, l, a) in carsDetected:
        cv2.rectangle(parkLine, (x, y), (x + l, y + a), (0, 0, 255), 2)

    for line in lines:
        for x, y, h, w in line:
                if (x == h and (y - w) > 450) or (y == w and (300 <= y <= 450)):
                    cv2.line(parkLine, (x, y), (h, w), (0, 255, 0), 2)

    cannyLines = cv2.addWeighted(park, 0.8, parkLine, 1, 0)
    cv2.imshow("Hough", cannyLines)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()