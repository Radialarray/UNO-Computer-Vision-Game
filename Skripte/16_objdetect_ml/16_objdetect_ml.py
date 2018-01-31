import cv2
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from operator import itemgetter
from os import listdir
from os.path import isfile, join, isdir

# ACHTUNG: Zusätzliche Voraussetzungen für dieses Sample für Machine Learning:
# SciKit + SciKit-Learn + SciKit-Image, Installation mit:
# pip install scipy sklearn

# Trainingsdaten-Pfad - für jedes Objekt das erkannt werden soll muss hier ein
# Unterordner mit entsprechenden Trainingsdaten abgelegt sein. Der Name des
# Unterordners ist gleichzeitig das Label was später bei der Erkennung verwendet wird
trainingDataFolder = "mlTrainingData"

# Mathematische "Dimensionalität" des ML-Algorithmus
dimensions=128


### AB HIER ML-TRAINING 
# Feature Descriptor initialisieren
# Dieser liefert ähnlich wie klassischen Featureerkenner eine Beschreibung des Bildinhalts
hog = cv2.HOGDescriptor()

data=[] # Array für HOG-Features
labels=[]

# Statusmeldung / Stoppuhr
print("Analyzing training data")
start_time = time.time()

# Für alle vorhandenen Trainingsdaten Features erkennen, speichern und Label (= in welche Klasse
# das Bild einzusortieren ist) zuweisen
for node in listdir(trainingDataFolder):
    #print("Listing "+node)
    if(isdir(join(trainingDataFolder,node))):
        #print(node+" is directory")
        for f in listdir(join(trainingDataFolder, node)):
            if isfile(join(trainingDataFolder, node, f)) and f != ".DS_Store":
                #print(f+" is file in "+join(trainingDataFolder, node, f))
                image = cv2.imread(join(trainingDataFolder, node, f))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                imgfeatures = cv2.resize(image, (dimensions,dimensions), interpolation = cv2.INTER_AREA)
                imgfeatures = hog.compute(imgfeatures)
                imgfeatures = np.squeeze(imgfeatures)
                data.append(imgfeatures)
                labels.append(node)

data = np.stack(data)

# Hier wird der ML-Algorithmus initialisiert und anschließend (clf.fit) trainiert
#clf=SVC(C=10000,kernel="linear",gamma=0.000001)
clf = KNeighborsClassifier(n_neighbors=1)

# Anmerkung: Je nach Use-Case macht es Sinn unterschiedliche Algorithmen zu verwenden. In diesem
# Beispiel bringen beide oben genannten Algorithmen (SVC und K-Nearest-Neighbour) ungefähr gleich
# gute Ergebnisse

# In "features" stecken die HOG-Features aus hoggify, Labels ist ein Array von Kennzeichnungen
# welche Bilder positiv (1) und negativ (0) sind. Es kann hier auch mit mehr als zwei Labels
# gearbeitet werden, der Algorithmus klassifiziert letztendlich immer in die Anzahl an Kategorien
# die beim Lernen verwendet wurden.
clf.fit(data,labels) 

# Statusmeldung / Stoppuhr
duration = (time.time()-start_time)
print("Analysis done in "+str(duration)+"sec")


### Ab hier: Klassifizierungs-Test mit 3 bisher unbekannten Bildern

# Man kann hier auch mit einem Webcam-Echtzeit-Feed arbeiten, allerdings muss dafür das Webcam-
# Bild vorverarbeitet werden so dass die eingehenden Frames den Trainingsdaten möglichst ähnlich sind.
# Oder man trainiert bereits mit Bildern von der Webcam (Tip: Video aufnehmen, Objekt vor der Kamera
# drehen und bewegen und die einzelnen Frames als Trainingsdaten nehmen).
# Wir testen hier nur mit 3 Bildern die nicht in den Trainingsdaten enthalten waren (wichtig!) aber
# trotzdem die trainierten Objekte zeigen.

image_apple = cv2.imread(trainingDataFolder+"/apple.jpg", 0)
image_mug = cv2.imread(trainingDataFolder+"/mug.jpg", 0)
image_banana = cv2.imread(trainingDataFolder+"/banana.jpg", 0)

image_apple_gray = cv2.resize(image_apple, (dimensions, dimensions), interpolation = cv2.INTER_AREA)
image_mug_gray = cv2.resize(image_mug, (dimensions, dimensions), interpolation = cv2.INTER_AREA)
image_banana_gray = cv2.resize(image_banana, (dimensions, dimensions), interpolation = cv2.INTER_AREA)

features_apple = hog.compute(image_apple_gray).T
features_mug = hog.compute(image_mug_gray).T
features_banana = hog.compute(image_banana_gray).T

print("Classification results:")
print("=======================")
print("Prediction for apple image: "+str(clf.predict(features_apple)))
print("Prediction for mug image: "+str(clf.predict(features_mug)))
print("Prediction for banana image: "+str(clf.predict(features_banana)))