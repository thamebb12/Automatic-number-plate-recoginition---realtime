from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor, ocr_image
import cv2
import numpy as np
import time
import serial
import threading
import pymongo
from datetime import datetime
import pytz

client = pymongo.MongoClient("mongodb+srv://thame:thame@thamefinalproject.gvb3usm.mongodb.net/?retryWrites=true&w=majority")

#def sleeper():
    #print("Thread starting")
    #time.sleep(10)
    #print("Thread ending")

#ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
def serial_communication(text):
        try:
            ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
            if ser.is_open:
                print("Serial port is open")
            else:
                print("Serial port is not open")
                return

        # Perform read or write operations here
            ser.write(b'180')
            time.sleep(8)
            ser.write(b'0')
            time.sleep(5)
            ser.close()
        # ...

        except serial.SerialException as e:
            print("Error:", e)
        finally:
            if ser.is_open:
                ser.close()
                print("Serial port closed")

    #ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
    #while True:
        #your code for sending data to the serial device
        #ser.write("180\r".encode())
        #time.sleep(10)
        #ser.write("0\r".encode())
        #ser.write(b'180')
        #time.sleep(8)
        #ser.write(b'0')
        #time.sleep(8)
        #ser.close()

model = YOLO("/home/peerawit/Desktop/Project/plate/v8/runs/detect/train/weights/best.pt")
files_number = 1
# initialize the webcam
cap = cv2.VideoCapture(0)

#processed = 0
#captured = False
while True:
    # capture a frame from the webcam
    ret, frame = cap.read()

    frame = cv2.resize(frame, (640, 480))

    # perform object detection with YOLO
    results = model(frame, conf=0.5)

    # iterate over the list of Result objects
    for res in results:

        # iterate over the Boxes object inside each Result object
        for i, box in enumerate(res.boxes):
            
            # extract the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # extract the cropped region of interest
            roi = frame[y1:y2, x1:x2]

            if roi is not None:
                # convert the image to grayscale
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # apply thresholding
                _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                 # perform distance transform
                dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)

                # scale the values in dist_transform
                dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                cv2.imwrite("/home/peerawit/Desktop/Project/plate/Crop_image/roi.jpg", roi)
                cv2.imwrite("/home/peerawit/Desktop/Project/plate/Crop_image/gray.jpg", gray)
                #cv2.imwrite("/home/peerawit/Desktop/Project/plate/plate_after_detect/plates1/roi.jpg", roi)
                cv2.imwrite("/home/peerawit/Desktop/Project/plate/Crop_image/dist_transform.jpg", dist_transform)
                #text = ocr_image(roi, (0, 0, roi.shape[1], roi.shape[0]))
                text = ocr_image(dist_transform, (0, 0, dist_transform.shape[1], dist_transform.shape[0]))
                files_number += 1
            #else:
                #text = ""
            # display the OCR result
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                if len(text) >= 6:
                    detect_plate = True

                    img_path = '/home/peerawit/Desktop/Project/plate/plate_after_detect/plates1/plate_' + str(files_number) + '.jpg'
                    cv2.imwrite(img_path, roi)
                    print('License plate image saved to:', img_path)

                    file_path = '/home/peerawit/Desktop/Project/plate/plate_after_detect/licens_plates1/plate_' + str(files_number) + '.txt'
                    with open(file_path, 'w') as f:
                        f.write(text)

                    db = client["ThameFinalProject"]
                    col = db["ProjectPlate"]
                    with open(file_path, "r") as f:
                        contents = f.read()

                    thai_tz = pytz.timezone("Asia/Bangkok")
                    timestamp = datetime.now(thai_tz)

                    doc = {
                        "text": contents,
                        #"timestamp": datetime.utcnow()
                        "timestamp": timestamp
                    }   
                    col.insert_one(doc)

                    # print a confirmation message
                    print("Text file inserted into MongoDB Atlas with timestamp")

                    serial_thread = threading.Thread(target=serial_communication, args=(text,))
                    serial_thread.start()
                    
                    print('License plate characters saved to:', file_path)
                else:
                    detect_plate = False

                # if text is detected in the license plate, pause for 8 seconds
                if detect_plate:
                    #serial_thread = threading.Thread(target=serial_communication, args=(text,))
                    #serial_thread.start()
                    time.sleep(10)
                    #thread = threading.Thread(target=sleeper)
                    #thread.start()
                    detect_plate = False

                #serial_thread = threading.Thread(target=serial_communication, args=(text,))
                #serial_thread.start()


                """
                if text != "":
                    ser.write("90\r".encode())
                    time.sleep(8)
                    ser.write("0\r".encode())
                """
    # display the frame
    cv2.imshow('License Plate Detection', frame)

    # exit on ESC key
    if cv2.waitKey(1) == 27:
        break
    
    #time.sleep(15)
    # reset the processed counter after each iteration

# release the webcam and close all windows
#cap.release()
#cv2.destroyAllWindows()