import cv2
import mediapipe as mp
import time

	

	

class FaceDetector():
	def __init__(self, minDetectionCon=0.5):
		self.minDetectionCon = minDetectionCon
		self.mpFaceDetection = mp.solutions.face_detection
		self.mpDraw = mp.solutions.drawing_utils
		self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon) #parameter helps in making it accurate

	def findFaces(self, img, draw=True):
		self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.faceDetection.process(imgRGB)
		#results.detections has id, bounding box dimensions, confidence score, coordinates of key points
		bboxs = []
		if self.results.detections:
			for id, detection in enumerate(self.results.detections):
			#mpDraw.draw_detection(img, detection) -> bounding box + key points are drawn
				bbox_from_class = detection.location_data.relative_bounding_box
				ih, iw, ic = img.shape
				bbox = int(bbox_from_class.xmin*iw), int(bbox_from_class.ymin*ih), \ int(bbox_from_class.width*iw), int(bbox_from_class.height*ih)
				bboxs.append([id, bbox, detection.score])

				if draw:
					img = self.drawBoundingBox(img, bbox)
		return img, bboxs

	def drawBoundingBox(self, img, bbox, l=30, thickness=5):
		x, y, w, h = bbox
		x1, y1 = x+w, y+h

		cv2.line(img, (x, y), (x+l, y), (255, 0, 255), thickness)
		cv2.line(img, (x, y), (x, y+l), (255, 0, 255), thickness)
		cv2.line(img, (x1, y), (x1-l, y), (255, 0, 255), thickness)
		cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), thickness)
		cv2.line(img, (x, y1), (x+l, y1), (255, 0, 255), thickness)
		cv2.line(img, (x, y1), (x, y1-l), (255, 0, 255), thickness)
		cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 255), thickness)
		cv2.line(img, (x1, y1), (x1, y-l), (255, 0, 255), thickness)
		return img 


def main():
	cap = cv2.VideoCapture(0)
	pTime = 0

	while True:
		success, img = cap.read()
		#frame rate (fps)
		cTime = time.time()
		fps = 1/(cTime - pTime)
		pTime = cTime
		cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

		cv2.imshow('Image', img)
		cv2.waitKey(1)


if __name__ == "__main__":
	main()
