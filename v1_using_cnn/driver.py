import numpy as np
import cv2
from getKeys import key_check
from screengrab import grab_screen
from input import Z, L, PressKey, ReleaseKey
from modelPred import predict
from preprocessing import img_preProcess
import time

def right():
    ReleaseKey(Z)
    PressKey(L)

def jump():
    ReleaseKey(L)
    PressKey(Z)

def rightJump():
    ReleaseKey(Z)
    PressKey(L)
    PressKey(Z)

do = True
def main():
    global do
    while True:
        frame = grab_screen([10, 40, 770, 700])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = frame[300:,100:]
        roi = cv2.resize(roi, (80,60))
        temp = roi
        roi = img_preProcess(roi)
        if do == True:
            prediction = predict(roi)
            prediction = prediction.argmax(axis=1)
            
            if prediction == 0:
                right()
            elif prediction == 1:
                jump()
            elif prediction == 2:
                rightJump()
        
            print(prediction)

        temp = cv2.resize(temp, (320,240))
        cv2.imshow('frame', temp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()