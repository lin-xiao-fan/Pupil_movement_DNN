import cv2
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 打开视频文件
video_path = r"C:\Users\tseng\Desktop\EyeTracking\2.0\曾X甯_錄影檔 (5)_ 剪_3.0.mp4"
cap = cv2.VideoCapture(video_path)

frame_counter = 0

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件.")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 保存每一帧为单独的图像文件
        filename = f"C:/Users/tseng/Desktop/EyeTracking/2.0/FRAME (5)_head/frame_{frame_counter}.jpg"
        cv2.imwrite(filename, frame)

        frame_counter += 1

    cap.release()

print(f"帧数 {frame_counter} 帧为图像.")

