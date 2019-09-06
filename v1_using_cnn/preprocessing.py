import numpy as np
def img_preProcess(frame):
    frame = frame/255
    frame = np.array(frame).reshape(-1, 80, 60, 1)
    return frame