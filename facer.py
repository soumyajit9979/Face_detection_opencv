import numpy as np
import cv2 as cv
capture=cv.VideoCapture(1)
haar_cascade=cv.CascadeClassifier('harr_face.xml')
# feature= np.load('feature.npy',allow_pickle=True)
# labels=np.load('labels.npy')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
people=['om','raghu','sumo']

while True:
    isTrue, frame=capture.read()
    cv.imshow('abc',frame)
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face_rect=haar_cascade.detectMultiScale(gray,1.1,5)
    for (x,y,w,h) in face_rect:
        face_roi=gray[x:x+h,y:y+h]

        label, confidence=face_recognizer.predict(face_roi)

        cv.putText(frame,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    # face_detect=haar_cascade.detectMultiScale(gray1,scaleFactor=1.1, minNeighbors=11)
    # print(len(face_detect))

    # for (x,y,w,h) in face_detect:
    #     cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    # cv.imshow('box',frame)
    cv.imshow('wqe',frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
# cv.waitKey(0)
cv.destroyAllWindows()