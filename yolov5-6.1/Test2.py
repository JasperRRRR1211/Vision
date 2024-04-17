import cv2

video = cv2.VideoCapture(1)
while True:
    # 从摄像头读取帧
    ret, frame = video.read()

    if not ret:
        break

    # 显示帧
    cv2.imshow('YOLOv5 Real-time Object Detection', frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
