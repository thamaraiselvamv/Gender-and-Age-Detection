# Advanced Gender & Age Detection (with Close-Window Support)

import cv2
import time
import argparse
import os
import urllib.request

# ---------------------------------------------------
# AUTO DOWNLOAD MODEL FILES
# ---------------------------------------------------

model_files = {
    "faceProto": "opencv_face_detector.pbtxt",
    "faceModel": "opencv_face_detector_uint8.pb",
    "ageProto": "age_deploy.prototxt",
    "ageModel": "age_net.caffemodel",
    "genderProto": "gender_deploy.prototxt",
    "genderModel": "gender_net.caffemodel",
}

urls = {
    "faceProto": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt",
    "faceModel": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector_uint8.pb",
    "ageProto": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
    "ageModel": "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel",
    "genderProto": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    "genderModel": "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel",
}

def download_models():
    print("Checking model files...")
    for key, file in model_files.items():
        if not os.path.exists(file):
            print(f"Downloading {file} ...")
            urllib.request.urlretrieve(urls[key], file)
    print("All model files ready!")

download_models()

# ---------------------------------------------------
# FACE DETECTION FUNCTION
# ---------------------------------------------------

def highlightFace(net, frame, threshold=0.7):
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False)

    net.setInput(blob)
    detections = net.forward()
    face_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            face_boxes.append([x1, y1, x2, y2, confidence])

    return frame_copy, face_boxes

# ---------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="Path to image or video")
args = parser.parse_args()

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

faceNet = cv2.dnn.readNet(model_files["faceModel"], model_files["faceProto"])
ageNet = cv2.dnn.readNet(model_files["ageModel"], model_files["ageProto"])
genderNet = cv2.dnn.readNet(model_files["genderModel"], model_files["genderProto"])

AGE_LABELS = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
GENDER_LABELS = ['Male', 'Female']
GENDER_COLORS = {"Male": (255, 0, 0), "Female": (255, 0, 255)}

# ---------------------------------------------------
# VIDEO / IMAGE SOURCE
# ---------------------------------------------------

video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20
prev_time = 0

# ---------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------

while True:
    ret, frame = video.read()
    if not ret:
        break

    # FPS
    new_time = time.time()
    fps = 1 / (new_time - prev_time) if prev_time else 0
    prev_time = new_time

    resultImg, faceBoxes = highlightFace(faceNet, frame)

    for (x1, y1, x2, y2, conf) in faceBoxes:
        face = frame[max(0, y1-padding): min(y2+padding, frame.shape[0]),
                     max(0, x1-padding): min(x2+padding, frame.shape[1])]

        # Gender prediction
        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )

        genderNet.setInput(blob)
        gender = GENDER_LABELS[genderNet.forward()[0].argmax()]

        ageNet.setInput(blob)
        age = AGE_LABELS[ageNet.forward()[0].argmax()]

        color = GENDER_COLORS[gender]
        cv2.rectangle(resultImg, (x1, y1), (x2, y2), color, 2)
        cv2.putText(resultImg, f"{gender}, {age}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show FPS
    cv2.putText(resultImg, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Advanced Age & Gender Detection", resultImg)

    # -----------------------------
    # WINDOW CLOSE SUPPORT
    # -----------------------------
    key = cv2.waitKey(1) & 0xFF

    # ESC key
    if key == 27:
        break

    # User clicked the close (X)
    if cv2.getWindowProperty("Advanced Age & Gender Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# SHUTDOWN CLEANLY
video.release()
cv2.destroyAllWindows()
