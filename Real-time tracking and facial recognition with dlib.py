# -*- coding: utf-8 -*-
"""
The following python script utilizes the dlib library for facial recognition.
 The script first reads in a database of faces stored in a saved CSV file. 
 The database contains face descriptors and bounding boxes for each face detected in a set of images.
Once the database has been loaded, the script then uses the dlib library to 
perform real-time facial recognition on a live stream from a PC webcam. 
The webcam captures images in real-time, and the dlib library is used to detect faces in each image 
and compare them to the faces in the database.

Since the detection process can be computationally expensive, we only make a detection once 
or twice in n frames and the rest of the n - 2 frames we perform object tracking which is 
made easier with dlib library, in order to reduce the compute cost and increase FPS throughput.


The script implements a brute nearest neighbor search algorithm to compare 
the faces detected in the webcam stream to the faces in the database. 
The algorithm calculates the distance between the face descriptors for each pair of faces
 and returns the closest match. 
 If a match is found, the script will display the name of the person in the database 
 that the live face is closest to.

This script provides a demonstration of how the dlib library can be used for
 real-time facial recognition and multiple objects tracking.
The use of a saved CSV file for storing the face database allows for a flexible and scalable solution, 
as the database can be easily updated or expanded as needed. 
"""
import time
from my_dlib_funcs import *

print(os.getcwd())



# Set working paths
database_path =  os.getcwd() + '/database'
test_path = os.getcwd() + '/testing imgs'
output_path = os.getcwd() +'/Outputs'

# Set models paths
shape_predictor_path = os.getcwd() + '/models/shape_predictor_68_face_landmarks_GTX.dat'
face_recognition_model_path = os.getcwd() + "/models/dlib_face_recognition_resnet_model_v1.dat"
cnn_model_path = os.getcwd() + '/models/mmod_human_face_detector.dat'

# Load the shape predictor and face recognition models

predictor = dlib.shape_predictor(shape_predictor_path)
face_rec = dlib.face_recognition_model_v1(face_recognition_model_path)

# Load the CNN and HOG face detection models

HOG_face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_model_path)


# get the reference face descriptors info from the people.csv file

filename = 'people_2.csv'

beg = time.time()
db_face_descriptors = read_db_csv(filename = filename)
print(time.time() - beg)


# display some useful info for debugging

print(len(db_face_descriptors))
print(type(db_face_descriptors))
print(type(db_face_descriptors[0]))

for desc in db_face_descriptors:
    print(desc.keys())
    print(desc["name"])
    print(desc["img path"])

print("********************************************************")


 # define a video capture object
cap = cv2.VideoCapture(0)
#cap.release()


width  = int(cap.get(3))  # float `width`
height = int(cap.get(4))  # float `height`


# setting up video output filename and path 
output_file = output_path + '/output_tracking.avi'

 # define an output VideoWriter  object
out = cv2.VideoWriter(output_file,
                      cv2.VideoWriter_fourcc(*"MJPG"), 
                      12,(width,height))

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error opening video stream or file")


# the recognition is performed k times for every skip frames
k = 2
skip = 30

# start i counter
i = -10
wait_cam = True

# Read the video frames
while cap.isOpened():
    
    i+= 1
    
    print("i = "+str(i))
# start recording time (to calculate FPS later)
    beg = time.time()

    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
        print("Error reading frame")
        break
      
    #print(frame.shape)

    # wait 10 frames for the webcam to warmup before the starting detections
    if (0>=i>=-10) and wait_cam:
        continue
    wait_cam = False
    
    if (k>=i>=1):
        print("**************************************************")
        trackers = []
        labels = []
        
        descriptors = get_face_descriptors(frame  ,                                            
                                           detection_scheme='HOG',                         
                                           shape_predictor = predictor, 
                                           #face_detector_path = cnn_model_path,
                                           face_recognizer = face_rec , 
                                           upsampling = 1)
        
        recognize(target_descriptors = descriptors,
                  database_descriptors = db_face_descriptors, 
                  max_dist_thresh = 0.7 )
        
        # get the details for each detected face in the frame i.e bounding boxes and name
        for desc in descriptors:
            print(len(descriptors))
            print(desc["name"])
            
            # get bounding box coordinates
            left = desc["bounding box"][0]
            top = desc["bounding box"][1]
            right = desc["bounding box"][2]
            bottom = desc["bounding box"][3]
            
            # put the face label and bounding box in the final ouput frame
            frame  = cv2.rectangle(frame ,(left,top),(right,bottom),(255,0,0),thickness = 4)
            frame  = cv2.putText(frame , desc["name"], (left - 5 ,top - 5), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # create a tracker for each detected object and start tracking 
            
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle( *desc["bounding box"])
            tracker.start_track(frame, rect)

            trackers.append(tracker)
            labels.append(desc["name"])
    else :
        for j,tracker in enumerate(trackers):
            
            tracker.update(frame)
            d = tracker.get_position()
        
            frame  = cv2.rectangle(frame ,(int(d.left()), int(d.top())),(int(d.right()), int(d.bottom())),
                                 color = (255,0,0),thickness = 4)
            
            frame  = cv2.putText(frame, labels[j] ,(int(d.left())-5, int(d.top())-5), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 
                                    fontScale = 1,thickness=2, color = (255,255,0))



    # display webcam stream with results        
    cv2.imshow("Output", frame )
    
    # calculate FPS
    end = time.time()
    fps = 1/(end - beg)
    out.write(frame)
    print(f'FPS = {fps:.2f}')
    
    # if i == skip : i = 0 we restart the counter
    i = i % skip

        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
out.release()
# Destroy all the windows
cv2.destroyAllWindows()
