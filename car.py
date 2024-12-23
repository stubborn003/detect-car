import cv2
from ultralytics import YOLO
import numpy as np

# 加载YOLO模型
model = YOLO('best.pt')

# 打开视频文件
cap = cv2.VideoCapture('TrafficPolice.mp4')

# 检查视频是否成功打开
if not cap.isOpened():
    print("错误：无法打开视频。")
    exit()

# 创建一个名为 'YOLO目标检测' 的窗口，并设定较小的初始窗口大小
cv2.namedWindow('YOLO Object Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO Object Detection', 1080, 540)  # 设置窗口初始大小为640x480

# 逐帧处理视频
while True:
    # 捕获每一帧
    ret, frame = cap.read()

    # 如果帧未成功捕获，则退出循环
    if not ret:
        break

    # 对当前帧进行预测
    results = model.predict(frame, classes=[0, 2])

    # 处理并在帧上绘制预测结果
    for result in results:
        frame_with_detections = result.plot(line_width=2, font_size=0.5)  # 较小的字体大小

    # 如果需要，将原始帧调整大小以匹配检测帧
    frame_resized = cv2.resize(frame, (frame_with_detections.shape[1], frame_with_detections.shape[0]))

    # 侧边并排连接原始帧和检测帧
    combined_frame = np.hstack((frame_resized, frame_with_detections))

    # 将组合帧的尺寸加倍
    combined_frame_doubled = cv2.resize(combined_frame, None, fx=2, fy=2)

    # 显示结果组合帧
    cv2.imshow('YOLO Object Detection', combined_frame_doubled)

    # 按 'q' 键时退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭所有OpenCV窗口
cap.release()
cv2.destroyAllWindows()