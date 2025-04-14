import cv2
import numpy as np
import os
import csv
from datetime import datetime
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.xml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
# iniciate id counter
id = 0
names = ['None', 'Rakshana', 'Uma', 'Prathiba', 'W']
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height
# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
import pandas as pd
now=datetime.now()
current_date=now.strftime("%Y-%m-%d")
try:
    f = pd.read_csv(current_date+'.csv',index_col=False)
except:
    f = pd.DataFrame(columns=['Name', 'Log Time', 'In/Out'])
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(
            img,
            str(id),
            (x + 5, y - 5),
            font,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            str(confidence),
            (x + 5, y + h - 5),
            font,
            1,
            (255, 255, 0),
            1
        )

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff  
    if k == 27:
        break
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
if id != 0:
    current_names = str(id)
    current_time = now.strftime("%H:%M:%S")
    if len(f[f['Name']==current_names]) % 2==0:
        log = 'In'
    else:
        log = 'Out'
    f.loc[len(f)]=[current_names,current_time, log]
    fname = current_date+'.csv'
    f.to_csv(fname,index=False)
