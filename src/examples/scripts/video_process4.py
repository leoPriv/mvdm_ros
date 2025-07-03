import cv2
import numpy as np

# ==== 1. 相机参数配置 ====
# 相机内参矩阵（假设640x480分辨率）
camera_matrix = np.array([[640, 0, 320],
                          [0, 640, 240],
                          [0, 0, 1]], dtype=np.float32)

# 畸变系数（桶形畸变）
dist_coeffs = np.array([-0.18, 0.02, 0, 0, 0], dtype=np.float32)  # 比之前更温和

# ==== 2. 透视变换参数 ====
frame_w, frame_h = 640, 480  # 目标分辨率

# 源点（扩大范围以保留边缘）
src_pts = np.float32([
    [30, frame_h - 30],  # 左下
    [30, 30],  # 左上
    [frame_w - 30, 30],  # 右上
    [frame_w - 30, frame_h - 30]  # 右下
])

# 目标点（保持原始宽高比）
dst_pts = np.float32([
    [0, frame_h],
    [0, 0],
    [frame_w, 0],
    [frame_w, frame_h]
])

# 计算透视变换矩阵
M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)

# ==== 3. 视频输入输出设置 ====
cap = cv2.VideoCapture("camera1_20250630_210334.mp4")
if not cap.isOpened():
    raise ValueError("无法打开视频文件！")

fps = cap.get(cv2.CAP_PROP_FPS)
actual_frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建两个VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_undistorted = cv2.VideoWriter("undistorted_output.mp4", fourcc, fps, (frame_w, frame_h))
out_birds_eye = cv2.VideoWriter("birds_eye_output.mp4", fourcc, fps, (frame_w, frame_h))

# ==== 4. 畸变校正映射 ====
# 保留更多边缘（alpha=1）
new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (frame_w, frame_h), alpha=1
)
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None,
    new_camera_matrix, (frame_w, frame_h), cv2.CV_16SC2
)

# ==== 5. 处理每一帧 ====
print("处理中... (按Q键可中断)")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 畸变校正
    undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    # 透视变换（鸟瞰图）
    birds_eye = cv2.warpPerspective(undistorted, M_persp, (frame_w, frame_h))

    # 保存两个视频
    out_undistorted.write(undistorted)
    out_birds_eye.write(birds_eye)

    # 实时预览（三合一画面）
    preview = np.hstack([
        cv2.resize(frame, (320, 240)),  # 原始帧
        cv2.resize(undistorted, (320, 240)),  # 校正帧
        cv2.resize(birds_eye, (320, 240))  # 鸟瞰图
    ])
    # 添加标签
    cv2.putText(preview, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(preview, "Undistorted", (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(preview, "Birds-Eye", (650, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Processing Preview", preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==== 6. 释放资源 ====
cap.release()
out_undistorted.release()
out_birds_eye.release()
cv2.destroyAllWindows()
print("✅ 处理完成！生成两个视频：")
print(f"1. 畸变校正视频: undistorted_output.mp4")
print(f"2. 鸟瞰图视频: birds_eye_output.mp4")