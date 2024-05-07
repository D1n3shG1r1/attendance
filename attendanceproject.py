import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#code is ok report , depends upon your camera resolution (620*480 recomended) and system hardware

path = 'ImagesAttendance'
images =[]
classNames = []
myList = os.listdir(path)

#print('myList')
#print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

#print('classNames')
#print(classNames)

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markAttendance(name):
    with open('attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            
encodeListKnown = findEncodings(images)

#print(encodeListKnown)
#print(len(encodeListKnown))
#print('Encoding complete')


#read image from web cam
#cap = cv2.VideoCapture(0)
#print('cap')
#print(cap)
cap = cv2.VideoCapture(1)
cap.set(3,640) # set Width
cap.set(4,480) # set Height


while True:
    success, img = cap.read()
    #print('img shape')
    #print(img)
    
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) #resize image
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) #convert to rgb
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    #print('facesCurFrame')
    #print(facesCurFrame)
    #print('encodesCurFrame')
    #print(encodesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchindex = np.argmin(faceDis)

        if matches[matchindex]:
            name = classNames[matchindex].upper()
            
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('camera',img)
    cv2.waitKey(1)