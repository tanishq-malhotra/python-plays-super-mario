import cv2
import numpy as np
import time

trainFile = 'training_data.npy'
cleaned_data_file = 'final_data_v1.npy'

#training_data = list(np.load(trainFile, allow_pickle=True))
cleaned_data = list(np.load(cleaned_data_file, allow_pickle=True))

# print(len(training_data))
""" frame = training_data[5000][0]
frame = cv2.resize(frame, (320,240))
cv2.imshow('frame',frame)
print(training_data[5000][1])
cv2.waitKey(0)
cv2.destroyAllWindows() """

""" for data in training_data:
    img = data[0]
    key = data[1]
    img = cv2.resize(img, (320, 240))
    cv2.imshow('frame', img)
    print(key)
    time.sleep(0.03)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break """

for data in cleaned_data:
        img = data[0]
        key = data[1]
        img = cv2.resize(img, (320, 240))
        cv2.imshow('frame', img)
        print(key)
        time.sleep(0.10)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


""" rj = 0
r = 0
j = 0
none = 0

for data in cleaned_data:
        img = data[0]
        key = data[1]

        if key == [1, 0, 0]:
                r += 1
        elif key == [0, 1, 0]:
                j += 1
        elif key == [0, 0, 1]:
                rj += 1
        elif key == None:
                none += 1

print('Length is {}, right: {}, jump: {}, right+jump: {}, None: {}'.format(len(cleaned_data),r,j,rj,none)) """