import tensorflow as tf 
from commonBlocks import darknet53,upsample,convolutional
import numpy as np
from load_weights import Load_weights
 

# hyperparameters 
NUM_CLASSES = 80
STRIDES = np.array([8,16,32])
ANCHORS =(1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875)
ANCHORS = np.array(ANCHORS).reshape(3,3,2)
weight_file = "./weights/yolov3.weights"


def yoloV3(input_layer):
	route_1, route_2, conv = darknet53(input_layer) 

	conv = convolutional(conv, (1,1,1024,512)) 
	conv = convolutional(conv,(3,3,512,1024)) 
	conv = convolutional(conv, (1,1,1024, 512))  
	conv = convolutional(conv, (3,3,512,1024))  
	conv = convolutional(conv,(1,1,1024,512)) 

	conv_lobj_branch = convolutional(conv,(3,3,512,1024))
	conv_lbbox = convolutional(conv_lobj_branch,(1,1,1024,3*(NUM_CLASSES+5)),
		activate= False, batch_norm = False)

	conv = convolutional(conv,(1,1,512,256))
	conv = upsample(conv)

	conv = tf.concat([conv, route_2], axis =-1) 
	conv = convolutional(conv,(1,1,768,256)) 
	conv = convolutional(conv,(3,3,256, 512))
	conv = convolutional(conv,(1,1,512,256))
	conv = convolutional(conv,(3,3,256,512))
	conv = convolutional(conv, (1,1,512,256))

	conv_mobj_branch = convolutional(conv, (3,3,256,512))
	conv_mbbox = convolutional(conv_mobj_branch ,(1,1,512,3*(NUM_CLASSES+5)),
		activate= False, batch_norm= False)


	conv = convolutional(conv, (1,1,256,128))
	conv = upsample(conv)

	conv = tf.concat([conv,route_1], axis = -1)

	conv = convolutional(conv, (1,1,384,128))
	conv = convolutional(conv, (3,3,128, 256))
	conv = convolutional(conv, (1,1,256, 128))
	conv = convolutional(conv, (3,3,128, 256))
	conv = convolutional(conv, (1,1,256, 128))

	conv_sobj_branch = convolutional(conv,(3,3,128, 256))
	conv_sbbox = convolutional(conv_sobj_branch,
		(1,1,256,3*(NUM_CLASSES+5)),activate= False , batch_norm= False)
	return [conv_sbbox, conv_mbbox, conv_lbbox]


def decode(conv_out, i = 0):
	conv_shape = tf.shape(conv_out)
	batch_size = conv_shape[0]
	output_size = conv_shape[1]

	conv_output = tf.reshape(conv_out, (batch_size, output_size,output_size, 3,5+NUM_CLASSES))
	
	conv_raw_dxdy = conv_output[:,:,:,:,0:2]
	conv_raw_dwdh = conv_output[:,:,:,:,2:4]
	conv_raw_conf = conv_output[:,:,:,:,4:5]
	conv_raw_prob = conv_output[:,:,:,:,5:]

	y = tf.tile(tf.range(output_size,dtype=tf.int32)[:,tf.newaxis],[1,output_size])
	x = tf.tile(tf.range(output_size, dtype= tf.int32)[tf.newaxis,:],[output_size,1])

	xy_grid = tf.concat([x[:,:,tf.newaxis],y[:,:,tf.newaxis]], axis = -1)
	xy_grid = tf.tile(xy_grid[tf.newaxis,:,:,tf.newaxis,:],[batch_size,1,1,3,1])
	xy_grid = tf.cast(xy_grid,tf.float32)

	pred_xy = (tf.sigmoid(conv_raw_dxdy)+xy_grid)*STRIDES[i]
	pred_wh = (tf.exp(conv_raw_dwdh)*ANCHORS[i])*STRIDES[i]
	pred_xywh = tf.concat([pred_xy,pred_wh], axis = -1)

	pred_conf = tf.sigmoid(conv_raw_conf)
	pred_prob = tf.sigmoid(conv_raw_prob)

	return tf.concat([pred_xywh, pred_conf, pred_prob], axis = -1)

def Model():
	input_layer = tf.keras.layers.Input([416,416,3])
	feature_maps = yoloV3(input_layer)

	bbox_tensors = []

	for i , fm in enumerate(feature_maps):
		bbox_tensor = decode(fm, i)
		bbox_tensors.append(bbox_tensor)


	model = tf.keras.Model(input_layer, bbox_tensors)
	model = Load_weights(model, weight_file)

	return model 







