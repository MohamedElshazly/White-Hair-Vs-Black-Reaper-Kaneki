from keras.models import load_model
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import applications
import cv2
import numpy as np


test_dir = '/home/mohamed/dir'

top_model = load_model('model.h5')

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory( test_dir,
		                                             target_size = (150, 150),
		                                             batch_size  = 1,
		                                             class_mode = 'binary')

preds = top_model.predict_generator(test_generator, len(test_generator.filenames))

bottle_model = load_model('bottle.h5')

sgd = optimizers.SGD(lr = 1e-4, momentum = 0.9, nesterov = True)

bottle_model.compile(loss='binary_crossentropy',
              optimizer= sgd,
              metrics=['accuracy'])



f_preds = bottle_model.predict(preds)
for i in range(len(test_generator.filenames)):
	if(np.around(f_preds[i]) == 0 ): 
		print("img num "+ str(i+1) +" is Black Reaper Kaneki")
	else:
		print("img num "+ str(i+1) +" is White Hair Kaneki")

