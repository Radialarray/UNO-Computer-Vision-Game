# UNO-Computer-Vision-Game

A computer vision project. Using python and cv2 as computer vision library.

The goal is to play the popular UNO cards game against a computer, which detects the cards in front of it's webcam.

We've made our own training data library of UNO cards.

Change to set up Computer Vision Python Workspace with: `source activate cvws2017`


At the moment is in working condition:
* color detection on single uno cards
* detection and filtering, where cards on the play area lay.
* cutting the detected cards for further processing
* feature detection with a machine learning approach
* returning detected card (quite accurate)
* saving the trained model for faster startup speed

Missing:
* UNO game logic
* color detection on filtered cards from webcam image



### How to setup the working environment for development?
Directly copied from the originator Florian Geißelhart

**_in german_**

=== CONDA installieren

Conda ist ein sogenannter Paketmanager für wissenschaftliche Anwendungen auf Basis von Python. OpenCV und viele andere wichtige Tools für Tracking und Computer Vision sind darin schon enthalten und verhältnismäßig einfach zu installieren.


1. Download Installscript unter https://conda.io/miniconda.html

2. chmod +x /Pfad/zu/Miniconda3-latest-MacOSX-x86_64.sh

3. ./Pfad/zur/Datei.sh

4. Installation ausführen / durchlaufen lassen


=== ENVIRONMENT erzeugen und aktivieren

Sogenannte Environments erlauben eine isolierte Umgebung zu erzeugen, die nicht von der Konfiguration des restlichen Rechners / Python-Systems abhängt, und diese auch nicht beeinflusst. Wir erzeugen uns ein eigenes Environment für diesen Kurs, in welchem wir später OpenCV etc. installieren (Befehl 1). Anschließend setzen wir dieses Environment als aktive Arbeitsumgebung (Befehl 2).

1. conda create --name=cvws2017 python=3.5 anaconda

2. source activate cvws2017

3. Um das Environment bei Bedarf wieder zu verlassen (Terminal schließen geht aber auch):
    source deactivate


=== PACKAGES installieren

Nachdem das Environment aktiv ist (zu erkennen am Name des Environments in Klammern, ganz vorne auf der Kommandozeile) müssen noch die Pakete für OpenCV etc in das Environment installiert werden.

1. pip install pillow
Pillow ist eine Softwarebibliothek die Python erweiterte Bildverarbeitungsfähigkeiten gibt

2. conda install -c https://conda.anaconda.org/menpo opencv3
Installiert OpenCV in Version 3.1.0

3. conda config --add channels conda-forge
Fügt dem Conda-Paketmanager eine neue Quelle für installierbare Software hinzu, in diesem Fall um auch dlib installieren zu können.

4. conda install dlib

5. Optional: Abschließender Test
        - Python aufrufen mit python
        - Folgende Befehle am Python-Eingabeprompt eingeben (ohne die >>>):
		>>> import cv2
		>>> import dlib
		>>> print(cv2.__version__)

(oder die template.py ausprobieren mit python /Pfad/zur/template.py)
