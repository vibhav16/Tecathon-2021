import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pytesseract
import re
from datetime import datetime
from PIL import Image
import face_recognition

pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'model'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)

while(True):
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)
    if frame_expanded.any():
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('ID CARD DETECTOR', frame)
        if((cv2.waitKey(1)%256 == 32)):
            cv2.imwrite('test_image.jpg', frame)
            img1 = cv2.imread('test_image.jpg')
            rgb_planes = cv2.split(img1)
            print(rgb_planes)
            result_planes = []
            result_norm_planes = []

            # Leaving below code for future work

            # for plane in rgb_planes:
            #     dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
            #     bg_img = cv2.medianBlur(dilated_img, 21)
            #     diff_img = 255 - cv2.absdiff(plane, bg_img)
            #     norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
            #                                                 dtype=cv2.CV_8UC1)
            #     result_planes.append(diff_img)
            #     result_norm_planes.append(norm_img)

            # result = cv2.merge(result_planes)
            # result_norm = cv2.merge(result_norm_planes)

            # # Remove noise
            # dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)
            text = pytesseract.image_to_string(img1).upper().replace(" ", "")
            print(text)
            date = str(re.findall(r"[\d]{1,4}[/-][\d]{1,4}[/-][\d]{1,4}", text)).replace("]", "").replace("[","").replace("'", "")
            print(date)
            born = date.split(",")[0]
            check_age = datetime.strptime(born, '%m/%d/%Y')
            age = (datetime.today() - check_age).days / 365
            print(age)
            if(age<21):
                print("Underage !!")
            if(age>21):
                print("Success !!")
           
            cv2.imshow('original',img1)
            cv2.imshow('edited',img1)
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.3, 7)
            for (x, y, w, h) in faces:
                ix = 0
                cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_color = img1[y:y + h, x:x + w]
                crop_pic_face = cv2.imwrite('crop_pic_face.jpg', roi_color)
                face_image = Image.open('crop_pic_face.jpg')
                check_image = Image.open('check.jpg')
                width, height = face_image.size
                newimage = check_image.resize((width, height))
                newimage.save('image_test.jpg')
                known_image = face_recognition.load_image_file("image_test.jpg")
                unknown_image = face_recognition.load_image_file("crop_pic_face.jpg")

                biden_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

                results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

                print(results)

                crop_pic = cv2.imshow('cropped_image', roi_color)
                print("success")
           
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

# Clean up
video.release()
cv2.destroyAllWindows()
