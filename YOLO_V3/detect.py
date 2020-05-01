from model import Model
import cv2 
from absl import logging
import time 
import tensorflow as tf
import utils 
import numpy as np  
import os 

# file of video 
videoFile = "./data/video7.mp4"
video_out = "converted.avi"


image_files = ("./data/image1.jpg","./data/image2.jpg")


def main(type = "image"):
	model = Model()
	names=utils.read_class_names("./data/classes.names")

	if type == "video" or type =="camera":
		if type == "video":
			vid = cv2.VideoCapture(videoFile) 
		elif type =="camera":
			vid = cv2.VideoCapture(0)

		width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = int(vid.get(cv2.CAP_PROP_FPS))
		codec = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(video_out, codec, fps, (width, height))
		while True: 
			_,img = vid.read()
			if img is None:
				logging.warning("Empty Frame")
				time.sleep(0.1)
				continue


			img_size = img.shape[:2]

			img_in = tf.expand_dims(img,0)
			img_in = utils.transform_images(img_in, 416)
			pred_bbox = model.predict(img_in)
			pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
			pred_bbox = tf.concat(pred_bbox, axis=0)

			boxes ,class_names, scores=utils.box_detector(pred_bbox)
			img=utils.drawbox(boxes ,class_names, scores,names,img)

			if video_out:
				out.write(img)
			img = cv2.resize(img, (1200, 700))
			cv2.imshow('output', img)

			if cv2.waitKey(1) == ord('q'):
				break

		cv2.destroyAllWindows()

	if type=="image":
		
		for i,image_file in enumerate(image_files):
			img=cv2.imread(image_file)
			img_in = tf.expand_dims(img,0)

			img_in = utils.transform_images(img_in,416)	
			pred_bboxes = model.predict(img_in)

			pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bboxes]
			
			pred_bbox = tf.concat(pred_bbox, axis=0)
			boxes ,class_names, scores=utils.box_detector(pred_bbox)
			img=utils.drawbox(boxes ,class_names, scores,names,img)
			img = cv2.resize(img, (1200, 700))
			cv2.imwrite("output_%d.jpg"%i,img)
			cv2.imshow('output7', img)

			if cv2.waitKey(0) == ord('q'):
				cv2.destroyAllWindows()



if __name__ == '__main__':
	main('video')
