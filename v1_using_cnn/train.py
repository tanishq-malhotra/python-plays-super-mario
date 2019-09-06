# importing libs
from model.minivgg import SmallVgg
from sklearn.metrics import classification_report
# imagedatagenerator used to do data augmantation
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np
import pickle

# defining parameters
modelPath = 'output/cnn.model'
dataPath = 'data'
lbName = 'lb.pickle'
labelBpath = 'output/'
trainFile = 'final_data_v1.npy'
width = 80
height = 60
depth = 1
classes = 3
LR = 0.01
EPOCHS = 75
# batch size
BS = 32
target_names = ['[1,0,0]', '[0,1,0]', '[0,0,1]']


print('loading data')
training_data = list(np.load(trainFile, allow_pickle=True))
print('data loaded')

# performing train test split
train = training_data[:-500]
test = training_data[-500:]

X_train = np.array([i[0] for i in train]).reshape(-1,width, height, 1)
y_train = np.array([i[1] for i in train])

X_test = np.array([i[0] for i in test]).reshape(-1,width, height, 1)
y_test = np.array([i[1] for i in test])

# converting the raw pixels from range 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255


# data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# compiling the model
model = SmallVgg.build(width=width,height=height,depth=depth,classes=classes)



# defining optimizer
opt = SGD(lr=LR,decay=LR/EPOCHS)
# as we have two classes so sparse_categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# training
print("Training the network")
M = model.fit_generator(aug.flow(X_train,y_train),validation_data=(X_test,y_test),
	steps_per_epoch = len(X_train)//BS, epochs=EPOCHS)

# Evaluating the model
pred = model.predict(X_test,batch_size=BS)
print(classification_report(y_test.argmax(axis=1),pred.argmax(axis=1), target_names=target_names))


# saving the model
print('\n'+'Saving the model')
model.save(modelPath)

print('Model is successfully trained and saved')