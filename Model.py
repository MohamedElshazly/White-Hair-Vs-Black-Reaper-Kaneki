import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras import applications
from keras import optimizers
import os
import cv2 

cwd = os.getcwd()
train_dir = '/home/mohamed/Desktop/Ken/train'
val_dir = '/home/mohamed/Desktop/Ken/val'
m1 = 80 
m2 = 30
epochs = 50 
b_size = 5
Height, Width = 150, 150


def model():

	model = applications.InceptionV3(include_top = False, weights = 'imagenet')

	train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True) 

	train_generator = train_datagen.flow_from_directory( train_dir,
		                                                 target_size = (Height, Width),
		                                                 batch_size  = b_size,
		                                                 class_mode = 'binary',
		                                                 shuffle = False) 

	val_datagen = ImageDataGenerator(rescale = 1./255)

	val_generator = val_datagen.flow_from_directory( val_dir,
		                                             target_size = (Height, Width),
		                                             batch_size  = b_size,
		                                             class_mode = 'binary',
		                                             shuffle = False)
	return train_generator, val_generator, model                                        	                                                  

   

train_gen, val_gen, model = model() 

model.save('model.h5')

print(model.summary())


def get_features(tg, vg, model): 

	train_features = model.predict_generator(tg, m1//b_size)
	np.save(open('train_features.npy', 'wb'), train_features)

	val_features = model.predict_generator(vg, m2//b_size)
	np.save(open('val_features.npy', 'wb'), val_features) 


def train(model):

	get_features(train_gen, val_gen, model)

	train_data = np.load(open('train_features.npy', 'rb'))
	train_labels = np.array([0] * (m1//2) + [1] * (m1//2))


	val_data = np.load(open('val_features.npy', 'rb'))
	val_labels = np.array( [0]*(m2//2) + [1]*(m2//2)) 


	final_model = Sequential() 

	final_model.add(Flatten(input_shape = train_data.shape[1:]))
	final_model.add(Dense(256))
	final_model.add(Activation('relu'))
	final_model.add(Dropout(0.5))
	final_model.add(Dense(1))
	final_model.add(Activation('sigmoid'))

	sgd = optimizers.SGD(lr = 1e-4, momentum = 0.9, nesterov = True)



	final_model.compile(optimizer = sgd,
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

	final_model.fit(    train_data,train_labels,
	                    batch_size = b_size,
	                    epochs = 50,
	                    validation_data = (val_data, val_labels))

	return final_model                     	

	                             
## running and saving the model 

bottle_model = train(model)

bottle_model.save('bottle.h5')


















