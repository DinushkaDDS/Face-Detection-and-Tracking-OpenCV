import numpy as np
import cv2
import argparse

#Variables for functionality in face detection.
scaleFactor = 1.05
gamma = 0.1
finalGamma = 0.1

#Function to get the Middle point of a box when top coordinate x , y and box width, height given.
def getMiddlePoint(a, b, c, d):
    return a+(c/2) , b + (d/2)

#Function to get the Top third point of a box when top coordinate x , y and box width, height given.
def getTwothirdPoint(a, b, c, d):
    return (a+c)/2 , b + (d/6)

#Function to maintain two points in same region
def maintainRange(frameX, frameY, boxX, boxY):
    if(frameX<boxX and frameY>boxY):
        cv2.putText(frame, "move bottom and left", (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    elif(frameX<boxX and frameY<boxY):
        cv2.putText(frame, "move left and top", (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    elif(frameX>boxX and frameY<boxY):
        cv2.putText(frame, "move right and top", (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    elif(frameX>boxX and frameY>boxY):
        cv2.putText(frame, "move right and bottom", (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    return
 
#function to correct the gamma values
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
	for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
    
    adjusted = cv2.LUT(image, table)
    return adjusted

#Function to find the optimal gamma value
def correctGamma(size):
    global gamma
    global finalGamma
    if(size < 1 ):
        gamma = gamma + 0.1
    else:
        finalGamma = gamma
        gamma = 0.1

    if(gamma > 2.5):
        finalGamma = 1
        gamma = 0.1

    return

#Function for calculate the focusing position of the frame
def getGoldenPosition(width, height):
    return width*2/3 , height/3

#Cascade file training for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


#Starting to capture frames using the front camera
cap = cv2.VideoCapture(0)
tracker = cv2.TrackerKCF_create()

#Getting frame sizes to analyze the photographic rules
frameWidth = cap.get(3)
frameHeight = cap.get(4)

while(True):
    
    ret, frame = cap.read()
    frame = adjust_gamma(frame, gamma)
    #cv2.imshow("Gamma",frame)
    faces = face_cascade.detectMultiScale(frame, scaleFactor,30)  
    correctGamma(len(faces))
    print(len(faces))
    
    if(len(faces)==1 or gamma == 0.1):
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Getting the camera frame Golden position    
frameX , frameY = getGoldenPosition(cap.get(3),cap.get(4))

while(True):
    ret, frame = cap.read()
    frame = adjust_gamma(frame, finalGamma)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor,30)

    bbox = (0,0,0,0)
    
    for (x,y,w,h) in faces:
        faceMiddle = getMiddlePoint(x,y,x+w,y+h)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        print("A face successfully detected!")

        bbox = (float(x),float(y),float(w),float(h))
        
    ok = True
    
    if(bbox ==(0,0,0,0)):
        print("Face not detected!")
    else:
        ok = tracker.init(frame, bbox)
        
        print("Tracker initiaized")
   
    
    #Loop for track the Identified Face
    while(bbox!=(0,0,0,0)):
        # Read a new frame
        ok, frame = cap.read()
        if not ok:
            print("Capturing error!")
            break
         
        # Start timer
        timer = cv2.getTickCount()
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            print("Tracking error!")
            tracker = cv2.TrackerKCF_create()
            break
 
            # Display FPS on frame
        #cv2.putText(frame, "FPS : " + str(int(fps)), (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        #calculating the tracking box mid/Golden position
        boxX , boxY = getMiddlePoint(bbox[0],bbox[1],bbox[2],bbox[3])

        #Calculating the required motion to capture the video in required manner
        maintainRange(frameX, frameY, boxX, boxY)
         
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if q pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #cv2.imshow("frame",frame)
        
    #Closing the camera connection
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
