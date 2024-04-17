import pytesseract
import NGYF as ng
import os
import cv2

video = cv2.VideoCapture('data/NGYF3.mov')
while video.isOpened():
        ret, frame = video.read()
        if not ret:
                break
        frame = ng.resize(frame, width=1280)
        cv2.imshow('video', frame)
        cv2.waitKey(1)
video.release()
cv2.destroyAllWindows()

'''
folder_path = 'runs/NGYFRUN/exp10/crops/Red'
#folder_path = 'runs/NGYFRUN/exp12/crops/Red'
for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        image = cv2.imread(filepath)
        ng.run(image)
        '''


