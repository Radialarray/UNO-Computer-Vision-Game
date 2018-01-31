import cv2
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# Trainiertes Model in Datei speichern
from sklearn.externals import joblib
from operator import itemgetter
from os import listdir
from os.path import isfile, join, isdir


# Mathematische "Dimensionalität" des ML-Algorithmus
dimensions=128


### AB HIER ML-TRAINING
# Feature Descriptor initialisieren
# Dieser liefert ähnlich wie klassischen Featureerkenner eine Beschreibung des Bildinhalts
hog = cv2.HOGDescriptor()

clf = joblib.load('feature_karten.pkl')

### Ab hier: Klassifizierungs-Test mit 3 bisher unbekannten Bildern

# Man kann hier auch mit einem Webcam-Echtzeit-Feed arbeiten, allerdings muss dafür das Webcam-
# Bild vorverarbeitet werden so dass die eingehenden Frames den Trainingsdaten möglichst ähnlich sind.
# Oder man trainiert bereits mit Bildern von der Webcam (Tip: Video aufnehmen, Objekt vor der Kamera
# drehen und bewegen und die einzelnen Frames als Trainingsdaten nehmen).
# Wir testen hier nur mit 3 Bildern die nicht in den Trainingsdaten enthalten waren (wichtig!) aber
# trotzdem die trainierten Objekte zeigen.

image_5_blau = cv2.imread("5_blau.jpg", 0)
image_9_gelb = cv2.imread("9_gelb.jpg", 0)

image_5_blau_gray = cv2.resize(image_5_blau, (dimensions, dimensions), interpolation = cv2.INTER_AREA)
image_9_gelb_gray = cv2.resize(image_9_gelb, (dimensions, dimensions), interpolation = cv2.INTER_AREA)

features_5_blau = hog.compute(image_5_blau_gray).T
features_9_gelb = hog.compute(image_9_gelb_gray).T

print("Classification results:")
print("=======================")
print("Prediction for 5 Blau image: "+str(clf.predict(features_5_blau)))
print("Prediction for 9 Gelb image: "+str(clf.predict(features_9_gelb)))
