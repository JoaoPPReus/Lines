import cv2
import numpy as np
import yaml

imageRead = "Imagens/shutterstock-9.png"
classifierCars = "Classificador/cascade3.xml"
dataCoordinatesVacancies = "Data\coordinates_vacancies.yml"
dataCoordinatesCars = "Data\coordinates_cars.yml"

SizeLine = 450
SizeVacancies = 100

vacancies = []
points = []
coordinates = []

park = cv2.imread(imageRead)
gray = cv2.cvtColor(park, cv2.COLOR_BGR2GRAY)

imgBlur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Blur", imgBlur)

imgCanny = cv2.Canny(imgBlur, 130, 130)
cv2.imshow("Cany", imgCanny)

lines = cv2.HoughLinesP(imgCanny, 0.01, (np.pi / 180), 15, np.array([]), 50, 50)

classifier = cv2.CascadeClassifier(classifierCars)
carsDetected = classifier.detectMultiScale(imgBlur, 1.01, 1, minSize=(30,30))

parkLine = np.copy(park) * 0

for line in lines:
    for x, y, h, w in line:
        if x == h and (y - w) > SizeLine:
            vacancies.append((x, y, SizeVacancies, int((w - y) / 2) + 5))
            coordinates.append(
                (x, y, x, y + int((w - y) / 2) + 5, x + SizeVacancies, y, x + SizeVacancies, y + int((w - y) / 2) + 5))

            vacancies.append((h, w, SizeVacancies, int((y - w) / 2) - 5))
            coordinates.append(
                (h, w, h, w + int((y - w) / 2) - 5, h + SizeVacancies, w, h + SizeVacancies, w + int((y - w) / 2) - 5))

i = 0
with open(dataCoordinatesCars, "w+") as data:
    for (x, y, l, a) in carsDetected:
        i += 1
        data.write("-\n          id: " + str(i) + "\n          coordinates: [" +
                   "[" + str(x + int(l / 2)) + "," + str(y + int(a / 2)) + "]]\n")

for (x, y, l, a) in vacancies:
    color = (0, 255, 0)
    with open(dataCoordinatesCars, "r") as dataCars:
        points = yaml.load(dataCars)

        if points is not None:
            for p in points:
                for z, w in np.array(p["coordinates"]):
                    if (((x <= z) and
                         (x + l >= z) and
                         (y + a <= w) and
                         (y >= w)) or

                        ((x <= z) and
                         (x + l >= z) and
                         (y <= w) and
                         (y + a >= w))):
                        color = (0, 0, 255)

    cv2.rectangle(parkLine, (x, y), (x + l, y + a), color, 2)

length = len(coordinates)
if coordinates is not None:
    with open(dataCoordinatesVacancies, "w+") as data:
        for tam in range(length):
            s = 0

            with open(dataCoordinatesCars, "r") as dataCars:
                points = yaml.load(dataCars)

                if points is not None:
                    for p in points:
                        for x, y in np.array(p["coordinates"]):
                            if (((coordinates[tam][2] <= x) and
                                 (coordinates[tam][4] >= x) and
                                 (coordinates[tam][3] <= y) and
                                 (coordinates[tam][5] >= y)) or
                                ((coordinates[tam][0] <= x) and
                                 (coordinates[tam][6] >= x) and
                                 (coordinates[tam][1] <= y) and
                                 (coordinates[tam][7] >= y))):
                                s = 1

            data.write("-\n          id: " + str(tam + 1) + "\n          status: " + str(s) +
                       "\n          coordinates: [" +
                       "[" + str(coordinates[tam][0]) + "," + str(coordinates[tam][1]) + "]," +
                       "[" + str(coordinates[tam][2]) + "," + str(coordinates[tam][3]) + "]," +
                       "[" + str(coordinates[tam][4]) + "," + str(coordinates[tam][5]) + "]," +
                       "[" + str(coordinates[tam][6]) + "," + str(coordinates[tam][7]) + "]]\n")

    with open(dataCoordinatesVacancies, "r") as data:
        points = yaml.load(data)

        if points is not None:
            for p in points:
                for x, y in np.array(p["coordinates"]):
                    color = (0, 255, 0)
                    if int(np.array(p["status"])) == 1:
                        color = (0, 0, 255)

                    cv2.circle(parkLine, (x, y), 5, color, -5)

cannyLines = cv2.addWeighted(park, 0.8, parkLine, 1, 0)
cv2.imshow("Hough", cannyLines)
cv2.waitKey(0)