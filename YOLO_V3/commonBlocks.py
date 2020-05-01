import tensorflow as tf 

class BatchNormalization(tf.keras.layers.BatchNormalization):
	def call(self, x, training= False):
		if not training:
			training = tf.constant(False)
		training = tf.logical_and(training, self.trainable)
		return super().call(x, training)




def convolutional(input_layer, filters_shape, down_sample = False,
		activate = True, batch_norm = True, regularization = 0.0005, reg_stddev = 0.01, activate_alpha = 0.1):

	if down_sample:
		input_layer = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(input_layer)
		padding ="valid"
		strides = 2
	else:
		padding ="same"
		strides = 1
	conv = tf.keras.layers.Conv2D(filters=filters_shape[-1],
		kernel_size = filters_shape[0],
		strides = strides,
		padding = padding,
		use_bias = not batch_norm,
		kernel_regularizer= tf.keras.regularizers.l2(regularization),
		kernel_initializer = tf.random_normal_initializer(stddev=reg_stddev),
		bias_initializer = tf.constant_initializer(0.)
		)(input_layer)

	if batch_norm:
		conv = BatchNormalization()(conv)
	if activate:
		conv = tf.nn.leaky_relu(conv, alpha= activate_alpha)

	return conv

def res_block(input_layer, input_channel, filter_num1, filter_num2):
	short_cut = input_layer
	conv = convolutional(input_layer, filters_shape=(1,1,input_layer,filter_num1))
	conv = convolutional(conv, filters_shape=(3,3,filter_num1,filter_num2))

	res_output = short_cut+ conv 
	return res_output

def darknet53(input_data):
	input_data = convolutional(input_data,(3,3,3,32))
	input_data = convolutional(input_data, (3,3,32,64), down_sample = True)

	for i in range(1):
		input_data = res_block(input_data, 64,32,64)

	input_data = convolutional(input_data, (3,3,64,128),down_sample=True)

	for i in range(2):
		input_data = res_block(input_data, 128,64,128)

	input_data = convolutional(input_data, (3,3,128,256), down_sample= True)

	for i in range(8):
		input_data = res_block(input_data,256,128,256)


	route_1 = input_data 

	input_data = convolutional(input_data,(3,3,256,512), down_sample= True)

	for i in range(8):
		input_data = res_block(input_data,512,256,512)
	route_2 = input_data
	input_data = convolutional(input_data,(3,3,512,1024), down_sample= True)

	for i in range(4):
		input_data= res_block(input_data,1024,512,1024)


	return route_1, route_2, input_data

def upsample(input_layer):
	return tf.image.resize(input_layer,(input_layer.shape[1]*2,input_layer.shape[2]*2),
		method='nearest')

