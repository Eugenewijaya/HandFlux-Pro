import cv2
import time

cap = cv2.VideoCapture(0)
print('opened', cap.isOpened())
for i in range(3):
    ret, frame = cap.read()
    print('iter', i, 'ret', ret, 'shape', None if frame is None else frame.shape)
    time.sleep(0.5)
cap.release()
