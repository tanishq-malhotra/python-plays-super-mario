#importing neccessary libs
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

class SmallVgg:

	def build(width,height,depth,classes):

		model = Sequential()
		# width and height are the inputs of the image and depth is the color layers of image
		# in our case its rgb so depth will be 3
		inputShape = (width,height,depth)
		# tensorflow uses channel_last so we use -1 otherwise we will be using 1
		chanDim = -1

		# defining layers of the model
		# Conv => relu => pool
		model.add(Conv2D(32,(3,3), padding='same', input_shape=inputShape))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		# 2(Conv => relu) => pool
		model.add(Conv2D(64,(3,3),padding='same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(64,(3,3), padding='same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=chanDim))

		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		# 3(Conv => relu) => pool
		model.add(Conv2D(128,(3,3),padding='same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(128,(3,3),padding='same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(128,(3,3), padding='same'))
		model.add(Activation('relu'))
		model.add(BatchNormalization(axis=chanDim))

		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		# fully connected layes => relu
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax activation
		# classes are the number to be classified(dogs or cats or flowers)
		model.add(Dense(classes))
		model.add(Activation('softmax'))

		return model