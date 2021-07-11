import cv2
print(cv2.__version__)
dispW = 640
dispH = 480
flip = 2

# for piCam use below camSet in VideoCapture function
# camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'

# for USB cam use below
cam = cv2.VideoCapture('/dev/video0')

#to save the output file
outVid = cv2.VideoWriter('videos/output.avi', cv2.VideoWriter_fourcc(*'XVID'), 21, (dispW,dispH))

# loading trained dataset for face detection
face_cascade = cv2.CascadeClassifier('/home/atakan/Desktop/codes/cascade/face1.xml')

# loading trained dataset for eye detection
eye_cascade = cv2.CascadeClassifier('/home/atakan/Desktop/codes/cascade/eye1.xml')

while True:
    ret, frame = cam.read()

    # to make computation easier turn image to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #all faces in the image is in faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # taking faces in a box
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 0, 255), 3) 
        cv2.rectangle(frame, (x,y), (x + w/2, y + h/2), (0, 0, 255), 3)
        # to make computation easier I created a region of interest 
        # from faces and converted it to gray scale again 
        # and checking for eyes
        roi_gray = gray[y:y+h, x:x+w] 

        #since I will put a box on colored image
        #I take that area from colored image as below
        roi_color = frame[y:y+h, x:x+w] 

        # all eyes in the image is in eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # taking eyes in a box
        for (xEye, yEye, wEye, hEye) in eyes:
            cv2.rectangle(roi_color, (xEye, yEye), (xEye + wEye, yEye + hEye), (0,255,0), 3)

    cv2.imshow('stereoCam',frame)
    cv2.moveWindow('stereoCam', 0,0)

    outVid.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
outVid.release()
cv2.destroyAllWindows()
    
