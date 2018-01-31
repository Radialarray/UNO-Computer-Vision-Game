# notwendige module importieren
import numpy as np # basis fuer opencv
import cv2 # opencv

# bild aus datei einlesen (dateiname entsprechend anpassen!)
img = cv2.imread('../2.jpg',0)


ret,thresh = cv2.threshold(img,127,255,0)
edges = cv2.Canny(img,100,200)
im2,contours,hierarchy = cv2.findContours(edges, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
area = cv2.contourArea(cnt)
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)



cv2.drawContours(img, [cnt], -1, (0,0,0), 3)

# eingelesenes bild anzeigen
cv2.imshow('image',img)

# auf tastatureingabe warten
k = cv2.waitKey(0)
#k = cv2.waitKey(0) & 0xFF # fuer 64bit

# tastatureingabe auswerten
if k == 27:         # ESC taste beendet das skript
    cv2.destroyAllWindows()
elif k == ord('s'): # taste 's' speichert das ergebnis in ein neues bild
    cv2.imwrite('image-output.png',img)
    cv2.destroyAllWindows()
