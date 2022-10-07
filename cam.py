import cv2 as cv

def rescaleFrame(frame):
    width=int(frame.shape[1]*4.5)
    height=int(frame.shape[0]*1)
    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    frame_resized=rescaleFrame(frame)
    
    cv.imshow('video',frame)
    cv.imshow('vid res',frame_resized)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
    
capture.release()
cv.destroyAllWindows()  