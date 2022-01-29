import cv2
import matplotlib.pyplot as plt
import numpy as np

from keras.models import model_from_json
import keras

persondict={0:"burak",1:"burkay",2:"yek",3:"kaleli"}
emodel_param = "/Users/caneralp/Downloads/Downloads/emodel.h5"
emodel = keras.models.load_model(emodel_param)


fruitdict={0:"Apple Braeburn",1:"Apple Granny Smith",2:"Apricot",3:"Avocado",4:"Banana",5:"Cantaloupe",6:"Cherry",7:"Clementine",8:"Corn",9:"Kiwi",10:"Lemon",11:"Onion White",
         12:"Orange",13:"Peach",14:"Pear",15:"Pepper Green",16:"Pepper Red",17:"Pineapple",18:"Plum",19:"Pomegranate",20:"Potato Red",21:"Strawberry",22:"Tomato",23:"Watermelon"}

fmodel_param = "/Users/caneralp/Downloads/Downloads/fmodel.h5"
fmodel = keras.models.load_model(fmodel_param)

def contours(frame):
    roi = frame[200:500, 900:1250]
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (13, 13), 5)
    dilat = cv2.dilate(blur, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    ret, thresh = cv2.threshold(dilatada, 100, 255, cv2.THRESH_BINARY)
    cnt, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnt) > 0:
        c = max(cnt, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        ((x, y), (w, h), rotation) = rect
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        cv2.drawContours(roi, [box], 0, (0, 255, 0), 2)

def dosya():
    file1 = open("/Users/caneralp/Downloads/Downloads/{}.txt".format(persondict[emaxindex]), "a")
    ali = list()
    for i in file1:
        ali.append(i[:-1])

    for j in ali:
        if ali.count(j) > 1:
            ali.remove(j)

    with open("/Users/caneralp/Downloads/Downloads/{}.txt".format(persondict[emaxindex]), "a") as file :
        for x in ali:
            file1.write(x + "\n")
    file1.close()


cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    roi = frame[200:500, 900:1250]
    cv2.rectangle(frame, (400, 100), (850, 600), (0, 255, 0), 2)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    num_faces = face_detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        if x<400 or y<100 or x+w>850 or y+h>600:
            cv2.putText(frame, "Yuzunu Yerlestiriniz", (800, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)
        else:
            eprediction = emodel.predict(roi)
            emaxindex = int(np.argmax(eprediction))
            cv2.putText(frame, "Yuz Icerde:Merhaba {},dosyan açılıyor".format(persondict[emaxindex]), (800, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2,cv2.LINE_AA)
            file = open("/Users/caneralp/Downloads/Downloads/{}.txt".format(persondict[emaxindex]), "a")
            cv2.rectangle(frame, (900, 200), (1250, 500), (0, 0, 255), 2)
            contours(frame)
            fprediction = fmodel.predict(roi)
            maxindex = int(np.argmax(fprediction))
            cv2.putText(frame, fruitdict[maxindex], (900, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_AA)
            flist=list()
            flist.append(fruitdict[maxindex])
            for j in flist:
                if flist.count(j) > 1:
                    flist.remove(j)
            for i in flist:
                file.write(i)

            file.close()



    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()