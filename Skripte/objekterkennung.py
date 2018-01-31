'''
Objekterkennung zwischen zwei Bildern

Erstellungsdatum: 14/12/2017
Name: Sven, Sarah, Moritz
'''

import cv2
import numpy as np


### Einstellungen

# Minimale Anzahl an matchenden Features fuer Erkennung
MIN_FEATURE_MATCH_COUNT = 10
# Maximale Featuredistanz (Qualitaet) fuer Erkennung
MAX_FEATURE_DISTANCE = 60


### Start und Setup

# Detektor und Matcher initialisieren
# ORB-Detektor, da OpenCV 3.x nur diesen unterstuetzt
detector = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
#bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# "Trainingsmodell" fuer Detektor generieren aus Vergleichsbild
# trainingKeypoints und trainingDescriptors enthalten danach Features des Trainingsbild
trainImg = cv2.imread("../Spielkarten/stopBlau.jpg")
trainingKeypoints = detector.detect(trainImg,None)
trainingKeypoints, trainingDescriptors = detector.compute(trainImg,trainingKeypoints)

cv2.drawKeypoints(trainImg,trainingKeypoints,trainImg,(0,255,0),2)
cv2.imshow("Search Object w/ Keypoints", trainImg)

compareImg = cv2.imread("../Spielkarten/1.jpg")
searchKeypoints = detector.detect(compareImg, None)
searchKeypoints, searchDescriptors = detector.compute(compareImg, searchKeypoints)
cv2.drawKeypoints(compareImg,searchKeypoints,compareImg,(0,255,0),2)
cv2.imshow("Search Object w/ Keypoints", compareImg)

### Webcam-Loop starten, Livevideo Bild fuer Bild verarbeiten
while True:


    # Matches zwischen Trainings- und Livebild finden
    matches = bf.match(searchDescriptors,trainingDescriptors)
    # Matches anhand Qualitaet / Distanz aussortieren
    goodMatch = []
    #print("Number of matches:"+str(len(matches)))
    for m in matches:
        #print (m.distance)
        if(m.distance < MAX_FEATURE_DISTANCE):
            goodMatch.append(m)

    ## Wenn nach Aussortieren genuegend Matches uebrig, erkanntes Objekt anzeigen
    if(len(goodMatch) > MIN_FEATURE_MATCH_COUNT):
        trainingPoints = []
        searchPoints = []
        # Paare von Featurepunkten in neues Array speichern
        for m in goodMatch:
            trainingPoints.append(trainingKeypoints[m.trainIdx].pt)
            searchPoints.append(searchKeypoints[m.queryIdx].pt)
        trainingPoints,searchPoints = np.float32((trainingPoints,searchPoints))

        # Anhand der Paare Homographie finden (=Transformation als Matrix gegenueber Originalbild)
        H,status = cv2.findHomography(trainingPoints,searchPoints,cv2.RANSAC,3.0)

        h,w,chs = trainImg.shape
        # Originalrechteck aus Trainingsdaten als Punkte in Array speichern
        trainBorder = np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        # Punkte gemaess H transformieren
        queryBorder = cv2.perspectiveTransform(trainBorder,H)
        # Als transformiertes 4-Polygon ins Bild einzeichnen
        cv2.polylines(compareImg, [np.int32(queryBorder)],True,(0,255,0),2)

        # Optional: Detektierte Keypoints zeichnen
        compareImg=cv2.drawKeypoints(compareImg,searchKeypoints,True,(0,255,0),2)
        # Optional: ROI/Objekt aus Webcam-Bild "geraderuecken" und in separates Bild ausgeben
        Hinv = np.linalg.inv(H)
        print("Genuegend Punkte gematcht- %d/%d"%(len(goodMatch),MIN_FEATURE_MATCH_COUNT))

    else:
        print("Nicht genuegend Punkte gematcht- %d/%d"%(len(goodMatch),MIN_FEATURE_MATCH_COUNT))
        compareImg = None

    # (Live-)Ausgabe der Bilder
    cv2.imshow('result',trainImg)
    #if(rightImg is not None):
    #    cv2.imshow('wrong', wrongImg)


    # Bei Tastatureingabe von q Schleife und damit Programm abbrechen
    if cv2.waitKey(1)==ord('q'):
        break

# Ressourcen freigeben
cv2.destroyAllWindows()
