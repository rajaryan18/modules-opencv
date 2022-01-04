import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75) #parameter helps in making it accurate

while True:
	success, img = cap.read()

	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = faceDetection.process(imgRGB)

	#results.detections has id, bounding box dimensions, confidence score, coordinates of key points
	if results.detections:
		for id, detection in enumerate(results.detections):
			#mpDraw.draw_detection(img, detection) -> bounding box + key points are drawn
			bbox_from_class = detection.location_data.relative_bounding_box
			ih, iw, ic = img.shape
			bbox = int(bbox_from_class.xmin*iw), int(bbox_from_class.ymin*ih), \ int(bbox_from_class.width*iw), int(bbox_from_class.height*ih)
			cv2.rectangle(img, bbox, (255, 0, 255), 2)

	cTime = time.time()
	fps = 1/(cTime - pTime)
	pTime = cTime
	cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

	cv2.imshow('Image', img)
	cv2.waitKey(1)
