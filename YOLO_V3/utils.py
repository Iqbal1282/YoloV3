import tensorflow as tf 
import numpy as np 
import cv2

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train

def box_detector(pred):


    
    center_x,center_y,width,height,confidence,classes = tf.split(pred,[1,1,1,1,1,-1], axis=-1)
    top_left_x=(center_x-width/2.)/ 416
    top_left_y = (center_y - height / 2.0)/416.0
    bottom_right_x = (center_x + width / 2.0)/416.0
    bottom_right_y = (center_y + height / 2.0)/416.0
    #pred = tf.concat([top_left_x, top_left_y, bottom_right_x,
    #                   bottom_right_y, confidence, classes], axis=-1)

    boxes = tf.concat([top_left_y,top_left_x,bottom_right_y,bottom_right_x],axis=-1)
    scores = confidence*classes
    scores = np.array(scores)

    scores = scores.max(axis=-1)
    class_index = np.argmax(classes, axis=-1)

    final_indexes = tf.image.non_max_suppression(boxes,scores, max_output_size= 20)
    final_indexes = np.array(final_indexes)
    class_names = class_index[final_indexes]
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_names = np.array(class_names)
    boxes = boxes[final_indexes,:]

    scores = scores[final_indexes]
    boxes = boxes*416

    return boxes ,class_names, scores

def drawbox(boxes, class_names,scores,names,img):
    data = np.concatenate([boxes,scores[:,np.newaxis],class_names[:,np.newaxis]],axis=-1)
    data = data[np.logical_and(data[:, 0] >= 0, data[:, 0] <= 416)]
    data = data[np.logical_and(data[:, 1] >= 0, data[:, 1] <= 416)]
    data = data[np.logical_and(data[:, 2] >= 0, data[:, 2] <= 416)]
    data = data[np.logical_and(data[:, 3] >= 0, data[:,3] <= 416)]
    data = data[data[:,4]>0.4]
    #print(data)

    img = cv2.resize(img, (416, 416))
    for i,row in enumerate(data):
        img=cv2.rectangle(img,(int(row[1]),int(row[0])),(int(row[3]),int(row[2])) ,(170, 1, 130),1)
        img = cv2.putText(img,(names[row[5]]+": "+"{:.4f}".format(row[4])),(int(row[1]),int(row[0])),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255,0 ),1)


    return  img
