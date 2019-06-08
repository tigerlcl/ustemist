#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Li Changlun, Tiger
# Email: tigerli@link.cuhk.edu.hk


#### Important Notice: Please install necessary packages manually at first.
from __future__ import absolute_import, division, print_function

import sys
sys.path.append("/Users/tigerli/SungemSDK-Python")
import os, cv2, time
import numpy as np
import hsapi.core as hs

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers, optimizers 
print("tensorflow version: ", tf.__version__)

# set-up global variables
ROI_ratio = 0.5
sample_rate = 0.5 # range(0-1)
label = ""
probi = 0
correct = None

WEBCAM = True # Set to True if use Webcam

def open_hs():
	print("##### Prepare the HornedSungem Hardware ###### ")
	# open_hs_device
	devices = hs.EnumerateDevices()
	if len(devices) == 0:
		print( "No devices found" )
		#quit()

	dev = hs.Device( devices[0] )
	dev.OpenDevice()

	return dev


print("##### Rebuild runtime for TensorFlow model ######")
# initial variables
feature_extractor_url="https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2"

def feature_extractor(x):
  feature_extractor_module = hub.Module(feature_extractor_url)
  return feature_extractor_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))
features_extractor_layer = layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE+[3])


# load CNN model that we trained for this project
model_path = os.getcwd()+'/my_model.h5'
# Recreate the exact same model, including weights and optimizer.
model = keras.models.load_model(model_path)
model.summary()

# initialize session
sess =  K.get_session()
init = tf.global_variables_initializer()
sess.run(init)

#compile model
model.compile(
	optimizer= tf.train.AdamOptimizer(), 
	loss='categorical_crossentropy',
	metrics=['accuracy'])


print("##### Ready for Image Classification #####")
# set labels
label_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic']

# Set camera mode
if WEBCAM: video_capture = cv2.VideoCapture(0)
else:
	dev = open_hs()

try:
	startTime = time.time()
	print("Press Ctrl + C to interrupt/stop the program")
	while True:
		if WEBCAM: _, image_raw = video_capture.read()
		else: image_raw = dev.GetImage(True) # Only get image

		# Crop ROI: Region Of Interest
		sz = image_raw.shape
		cx = int(sz[0]/2)
		cy = int(sz[1]/2)
		ROI = int(sz[0]*ROI_ratio)
		#ROI_y = int(sz[1]*ROI_ratio)
		cropped = image_raw[cx-ROI:cx+ROI,cy-ROI:cy+ROI,:]
		
		# Preprocess
		cropped = cropped.astype(np.float32)
		cropped[:,:,0] = (cropped[:,:,0] - 104)
		cropped[:,:,1] = (cropped[:,:,1] - 117)
		cropped[:,:,2] = (cropped[:,:,2] - 123)

		# video frame rate control
		nowTime = time.time()
		if(nowTime - startTime > sample_rate):
			# image classificaiton
			img = cv2.resize(cropped,tuple(IMAGE_SIZE)).astype(np.float16)
			img = np.expand_dims(img,axis=0)
			result = model.predict(img)
			probi, label = np.max(result), label_names[np.argmax(result)]
			correct = True if probi >= 0.3 else False
			#print("It is most likely to be {} with probability: {:.4f}".format(label, probi))

			#reset time
			startTime = time.time()
		
		#visualization
		if correct:
			cv2.putText(image_raw, "%s %0.2f %%" % (label, probi*100), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
		else:
			cv2.putText(image_raw, "CANNOT Classify due to LOW Accuracy" , (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (204, 0, 204), 2)
		cv2.rectangle(image_raw, (cy-ROI, cx-ROI), (cy+ROI, cx+ROI),(255,0,127), 5)
		cv2.imshow('Real-Time Image Classification',image_raw)
		

		key = cv2.waitKey(1)
		if key == ord('w'):
			ROI_ratio += 0.1
		elif key == ord('s'):
			ROI_ratio -= 0.1
		elif key == ord('l'):
			sample_rate += 0.1
		elif key == ord('h'):
			sample_rate -= 0.1

		if ROI_ratio < 0.1:
			ROI_ratio = 0.1

		if sample_rate < 0:
			sample_rate = 0

finally:
	print("bye bye")
	if not WEBCAM: dev.CloseDevice()

