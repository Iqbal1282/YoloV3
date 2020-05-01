import numpy as np 


def Load_weights(model,weight_file):

	wf = open(weight_file, 'rb')

	major , minor, revision , seen, _ = np.fromfile(wf,dtype= np.int32, count=5)
	j=0
	for i in range(75):
		conv_layer_name = 'conv2d_%d' %i if i>0 else 'conv2d'
		bn_layer_name = 'batch_normalization_%d' %j if j>0 else 'batch_normalization' 

		conv_layer = model.get_layer(conv_layer_name)
		filters = conv_layer.filters
		k_size = conv_layer.kernel_size[0]
		in_dim = conv_layer.input_shape[-1]


		if i not in [58,66,74]:
			# darknet weights: [beta, gamma, mean, variance]
			bn_weights = np.fromfile(wf, dtype= np.float32, count = 4*filters)
			bn_weights = bn_weights.reshape((4,filters))[[1,0,2,3]]
			bn_layer = model.get_layer(bn_layer_name)

			j+=1

		else:
			conv_bias = np.fromfile(wf,dtype= np.float32, count= filters)

		# darknet shape is (out_dim, in_dim, height,width)
		conv_shape = (filters, in_dim,k_size,k_size)
		conv_weights = np.fromfile(wf,dtype= np.float32, count= np.product(conv_shape))

		#tf shpae (height, width, in_dim, out_dim)
		conv_weights = conv_weights.reshape(conv_shape).transpose([2,3,1,0])


		if i not in [58,66,74]:
			conv_layer.set_weights([conv_weights])
			bn_layer.set_weights(bn_weights)
		else:
			conv_layer.set_weights([conv_weights,conv_bias])

	assert len(wf.read(0))==0, 'failed to read all data'
	wf.close()

	return model




