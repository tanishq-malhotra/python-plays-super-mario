import cv2
import numpy as np
import pandas as pd
from random import shuffle
from collections import Counter
trainFile = 'training_data.npy'
training_data = list(np.load(trainFile, allow_pickle=True))

df = pd.DataFrame(training_data)

print(Counter(df[1].apply(str)))
print('\n')

jump, right, rl = [], [], []
shuffle(training_data)
for data in training_data:
    img, key = data[0], data[1]

    if key == [1, 0, 0]:
        right.append(data)
    elif key == [0, 1, 0]:
        jump.append(data)
    elif key == [0, 0, 1]:
        rl.append(data)

right = right[:len(jump)]
rl = rl[:len(jump)]

final_data = right + jump + rl
shuffle(final_data)
np.save('final_data_v1.npy', final_data)

df = pd.DataFrame(final_data)
print(Counter(df[1].apply(str)))