import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
capture=cv.VideoCapture(0)
har_cascade=cv.CascadeClassifier("harr_face.xml")
people= ['sumo','raghu','om']
DIR = r'C:\Users\Raghuvir\Desktop\new\face'
features=[]
labels=[]
def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label=people.index(person)
        
        
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array= cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            
            faces_react=har_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=11)
            for (x,y,w,h) in faces_react:
                faces_roi=gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

features = np.array(features)
labels=np.array(labels)


face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)
face_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)


            
        
while True:
    isTrue, frame=capture.read()
    #cv.imshow('Biebs1',frame)
    edge=cv.Canny(frame,50,70)
    # cv.imshow('Biebs2',edge)
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    # #cv.imshow("Biebs5",gray)
    # threshold, thresh=cv.threshold(gray,125,255,cv.THRESH_BINARY)
    # #cv.imshow('Biebs4',thresh)
    # #contours, hierarchies=cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    # blank=np.zeros(frame.shape[:2],dtype=('uint8'))
    # #cv.drawContours(blank,contours,-1,(0,0,255),1)
    # #cv.imshow('Biebs3',blank)
    
    # mask=cv.circle(blank,(frame.shape[1]//2,frame.shape[0]//2),100,255,-1)
    # masked=cv.bitwise_and(edge,edge,mask=mask)
    # #cv.imshow('msked',masked)
    # adap=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,7,2)
    # # cv.imshow('adap',adap)
    # adap2=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,7,2)
    # # cv.imshow('adap2',adap2)
    # lap=cv.Laplacian(gray,cv.CV_64F)
    # lap=np.uint8(np.absolute(lap))
    # # cv.imshow('lap
    har_cascade=cv.CascadeClassifier("harr_face.xml")
    face_react= har_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=11)
    print(f'No. of faces = {len(face_react)}')
    for (x,y,w,h) in face_react:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)

    cv.imshow('show',frame)
    
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
cv.waitKey(0)
