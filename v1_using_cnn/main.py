import cv2
import numpy as np
from input import PressKey, Z, X, L, J, ReleaseKey
from screengrab import grab_screen
from encoding import oneHotEncoding
from getKeys import key_check

""" Right+JUMP(L+Z) = [0,0,1]
RIGHT (L) = [1,0,0]
JUMP (Z) = [0,1,0] """



while True:
    screen = grab_screen([10, 40, 770, 700])
    #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    roi = screen[300:,100:]
    roi = cv2.resize(roi, (80,60))
    cv2.imshow('frame', screen)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cv2.destroyAllWindows()