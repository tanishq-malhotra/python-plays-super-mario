import cv2
from screengrab import grab_screen
from getKeys import key_check
from encoding import oneHotEncoding
import numpy as np
import os
import time


# training file name
trainFile = 'training_data.npy'
# if training file exists then we load it else we start from begining
if os.path.isfile(trainFile):
    print('training file existed. Loading training_data')
    training_data = list(np.load(trainFile, allow_pickle=True))
else:
    print('No training data existed, creating new one')
    training_data = []


# main function
def main():
    # countdown of 5
    for i in range(6)[::-1]:
        print('Stating in {}'.format(i))
        time.sleep(1)

    initial_time = time.time()

    # main loop
    while True:
        # grabbing the screen
        screen = grab_screen([10, 40, 770, 700])
        # converting to gray
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        # getting the region of interest
        roi = screen[300:,100:]
        # resizing the frame to (80, 60)
        roi = cv2.resize(roi, (480, 270))
        # checking the keystroke
        key = key_check()
        # getting the one hot encoding of the key
        key = oneHotEncoding(key)
        # appending the frame with particular key stoke
        training_data.append([roi, key])

        print('Fps: {}'.format(time.time() - initial_time))
        initial_time = time.time()


        # if length of training data is 500, we stop and save
        if len(training_data) % 200 == 0:
            print('Length is {}, now saving the data'.format(len(training_data)))
            np.save(trainFile, training_data)

if __name__ == '__main__':
    main()