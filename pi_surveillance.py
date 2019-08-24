# import packages

#from /usr/home/pi/Documents/Motion_Detection/pyimagesearch import TempImage.jpg 
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
import dropbox
import imutils
import json
import time
import cv2
import inspect

# parse arguments

ap  = argparse.ArgumentParser()
ap.add_argument("p_name", type=str, help="path to ther JSON configuration file")
args = vars(ap.parse_args())


# filter warnings, load configuration

warnings.filterwarnings("ignore")
with open(args["p_name"]) as f:
    conf = json.load(f)
client = None

# Initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0
count_val = 0

# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab NumPy array representing image
    frame = f.array
    timestamp = datetime.datetime.now()
    text = "Unoccupied"
    
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21),0)
    
    # if the average frame is None, initialize it
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        rawCapture.truncate(0)
        continue
    
    
    # accumulate weighted averages between current frame and
    # previous frames, then compute difference between current
    # frame and the running average
    
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    
    # threshold the delta image, dilate the threshold image to fill
    # in holes, then find contours on threshold image
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, \
                           cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, \
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # loop over the contours
    for c in cnts:
        # ignore small contours
        if cv2.contourArea(c) < conf["min_area"]:
            continue
        
        # compute bounding box for contour and draw it
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0),2)
        text = "Occupied"
        
    # draw the text and timestamp
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, "Room Status: {}".format(text), (10,20), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, \
                0.35, (0,0,255),1)
    
    # check to see if the room is occupied
    if text == "Occupied":
        # check to see if enough time has passed between uploads
        if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
            # increment the motion counter
            motionCounter += 1
            
            # check to see if the number of frames with consistent motion
            # is high enough
            
            if motionCounter >= conf["min_motion_frames"]:
                # check to see if dropbox should be used
                count_val += 1
                if not(conf["use_dropbox"]):
                    # write images to a temporary file
##                    t = TempImage()
##                    cv2.imwrite(t.path, frame)
##                    # upload image to dropbox
##                    print("[UPLOAD] {}".format(ts))
##                    path = "/{base_path}/{time_stamp}.jpg".format(\
##                        base_path = conf["dropbox_base_path"], timestamp = ts)
##                    client.files_upload(open(t.path, "rb").read(), path)
##                    t.cleanup()
                    f_name = "motion{}.jpg".format(count_val)
                    print(f_name)
                    fold_name = "/home/pi/Documents/Motion_Detection/detected"
                    fold_name = "{fold_name}/{f_name}.jpg".format(fold_name \
                                                                  = fold_name, f_name = f_name)
                    cv2.imwrite(fold_name, frame)
                    
                # update the last uploaded timestamp and reset 
                lastUploaded = timestamp
                motionCounter = 0

    # otherwise, the room is not occupied
    else:
        motionCounter = 0
    
    # check to see if the frames should be displayed to screen
    if conf["show_video"]:
        # display to the security feed
        cv2.imshow("Security Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # if the q key is pressed
        if key == ord("q"):
            break
        
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
# print(inspect.getsource(PiCamera.capture_continuous))
