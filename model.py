## Use the previous model of Tracknet from https://gitlab.nol.cs.nycu.edu.tw/open-source/TrackNet/-/blob/master/Code_Python3/TrackNet_Three_Frames_Input/Models/TrackNet.py?ref_type=heads
# and TrackNet3 from https://gitlab.nol.cs.nycu.edu.tw/open-source/TrackNetv2/-/blob/master/3_in_3_out/TrackNet3.py?ref_type=heads
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras import ops
import tensorflow as tf
import cv2

def TrackNet(n_classes, input_height, input_width): # input_height = 360, input_width = 640

	imgs_input = Input(shape=(9,input_height,input_width))

	#layer1
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(imgs_input)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer2
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer3
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer4
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer5
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer6
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer7
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer8
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer9
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer10
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#layer11
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer12
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer13
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer14
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer15
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer16
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer17
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer18
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer19
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer20
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer21
	x = ( UpSampling2D( (2,2), data_format='channels_first'))(x)

	#layer22
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer23
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#layer24
	x =  Conv2D( n_classes , (3, 3) , kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	out_shape = Model(imgs_input , x ).output_shape
	#layer24 output shape: 256, 360, 640

	OutputHeight = out_shape[2]
	OutputWidth = out_shape[3]

	#reshape the size to (256, 360*640)
	x = (Reshape((-1, OutputHeight*OutputWidth)))(x)

	#change dimension order to (360*640, 256)
	x = (Permute((2, 1)))(x)

	#layer25
	gaussian_output = (Activation('softmax'))(x)
	# print("gaussian_output.shape:", gaussian_output.shape)
	model = Model( imgs_input , gaussian_output)
	model.outputWidth = OutputWidth
	model.outputHeight = OutputHeight

	# model.summary()

	return model

def TrackNet2( input_height, input_width ): # Originally input_height = 360, input_width = 640

	imgs_input = Input(shape=(9,input_height,input_width))

	#Layer1
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(imgs_input)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer2
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x1 = ( BatchNormalization())(x)

	#Layer3
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x1)

	#Layer4
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer5
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x2 = ( BatchNormalization())(x)
	#x2 = (Dropout(0.5))(x2) 

	#Layer6
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x2)

	#Layer7
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer8
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer9
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x3 = ( BatchNormalization())(x)
	#x3 = (Dropout(0.5))(x3)

	#Layer10
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x3)

	#Layer11
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer12
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer13
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
	#x = (Dropout(0.5))(x)

	#Layer14
	x = UpSampling2D( (2,2), data_format='channels_first')(x)

	#Layer15
	x = concatenate( [x, x3], axis=1)

	#Layer16
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer17
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer18
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
	
	#Layer19
	x = UpSampling2D( (2,2), data_format='channels_first')(x)

	#Layer20
	x = concatenate( [x, x2], axis=1)

	#Layer21
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer22
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer23
	x = UpSampling2D( (2,2), data_format='channels_first')(x)

	#Layer24
	x = concatenate( [x, x1], axis=1)

	#Layer25
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer26
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer27
	# x =  Conv2D(3,(1,1), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x =  Conv2D(1,(1,1), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	
	x = ( Activation('sigmoid'))(x)
	
	print ("layer27_x.shape:",x.shape)

	o_shape = Model(imgs_input , x ).output_shape

	filters = o_shape[1]
	OutputHeight = o_shape[2]
	OutputWidth = o_shape[3]
	
	#reshape the size 
	# x = (Reshape((-1, OutputHeight*OutputWidth )))(x) # For SparceCategoricalCrossEntropy(SCCE) loss
	# #change dimension order to (360*640, 256)
	# x = (Permute((2, 1)))(x)

	# x = (Reshape((OutputWidth*OutputHeight, filters)))(x) # Change to one dimension (640*360, 3), the best result for 3-frames-out, 0.99 accuracy for 1-frame-out 
	x = (Reshape((OutputWidth*OutputHeight, -1)))(x) # Change to one dimension (640*360, 3), the best result for 3-frames-out, 0.99 accuracy for 1-frame-out 


	model = Model( imgs_input , x)
	model.outputWidth = OutputWidth
	model.outputHeight = OutputHeight

	#Show model's details
	# model.summary()

	return model

def TrackNetU(n_classes, input_height, input_width):
	"""" Build the TrackNet model with U-net architecture 
	"""
	imgs_input = Input(shape=(9,input_height,input_width))

	# Contracting path
	#layer1
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(imgs_input)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer2
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x1 = ( BatchNormalization())(x)

	#Layer3
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x1)

	#Layer4
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer5
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x2 = ( BatchNormalization())(x)
	#x2 = (Dropout(0.5))(x2) 

	#Layer6
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x2)

	#Layer7
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer8
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer9
	x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x3 = ( BatchNormalization())(x)
	#x3 = (Dropout(0.5))(x3)

	#Layer10
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x3)

	#Layer11
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer12
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer13
	x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
	#x = (Dropout(0.5))(x)

	#Layer14
	x = UpSampling2D( (2,2), data_format='channels_first')(x)

	#Layer15
	x = concatenate( [x, x3], axis=1)

	#Layer16
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer17
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer18
	x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
	
	#Layer19
	x = UpSampling2D( (2,2), data_format='channels_first')(x)

	#Layer20
	x = concatenate( [x, x2], axis=1)

	#Layer21
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer22
	x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer23
	x = UpSampling2D( (2,2), data_format='channels_first')(x)

	#Layer24
	x = concatenate( [x, x1], axis=1)

	#Layer25
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer26
	x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer27
	x =  Conv2D( n_classes , (3, 3) , kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

        

	o_shape = Model(imgs_input , x ).output_shape

	OutputHeight = o_shape[2]
	OutputWidth = o_shape[3]

	#reshape the size to (256, 360*640)
	x = (Reshape((  -1  , OutputHeight*OutputWidth   )))(x)

	#change dimension order to (360*640, 256)
	x = (Permute((2, 1)))(x)

	#layer28
	gaussian_output = (Activation('softmax'))(x)

	model = Model( imgs_input , gaussian_output)


	model.outputWidth = OutputWidth
	model.outputHeight = OutputHeight

	#Show model's details
	#model.summary()

	return model



	



