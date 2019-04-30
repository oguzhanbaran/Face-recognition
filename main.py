import cv2
import numpy as np

kamera=cv2.VideoCapture(0)
while True:
    ret,kare=kamera.read()

    yuz_casc=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_alt.xml")
    resim_gri=cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)
    yuzler=yuz_casc.detectMultiScale(resim_gri,1.1,4)
    for (x,y,w,h) in yuzler:
        cv2.rectangle(kare,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Resim",kare)
    cv2.waitKey(1)
cv2.destroyAllWindows()