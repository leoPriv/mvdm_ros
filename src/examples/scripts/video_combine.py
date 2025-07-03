import cv2
import numpy as np


def process_and_combine_videos(video1_path, video2_path, output_path,
                               camera_params1, camera_params2,
                               size1=(640, 480), size2=(640, 480)):
    """
    处理并拼接两个视频
    :param camera_params1: 第一个相机的参数 (camera_matrix, dist_coeffs, persp_src, persp_dst)
    :param camera_params2: 第二个相机的参数
    :param size1: 第一个视频的最终输出尺寸
    :param size2: 第二个视频的最终输出尺寸
    """
    # 解包相机参数
    cam_mtx1, dist1, src_pts1, dst_pts1 = camera_params1
    cam_mtx2, dist2, src_pts2, dst_pts2 = camera_params2

    # 打开视频文件
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # 获取视频信息（以第一个视频为准）
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = size1[0] + size2[0]
    height = max(size1[1], size2[1])

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 为两个视频初始化矫正映射
    new_cam_mtx1, _ = cv2.getOptimalNewCameraMatrix(cam_mtx1, dist1, size1, alpha=1)
    map1_1, map1_2 = cv2.initUndistortRectifyMap(cam_mtx1, dist1, None, new_cam_mtx1, size1, cv2.CV_16SC2)
    persp_mtx1 = cv2.getPerspectiveTransform(src_pts1, dst_pts1)

    new_cam_mtx2, _ = cv2.getOptimalNewCameraMatrix(cam_mtx2, dist2, size2, alpha=1)
    map2_1, map2_2 = cv2.initUndistortRectifyMap(cam_mtx2, dist2, None, new_cam_mtx2, size2, cv2.CV_16SC2)
    persp_mtx2 = cv2.getPerspectiveTransform(src_pts2, dst_pts2)

    print("处理中... (按Q键可中断)")
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # 处理第一个视频帧
        undist1 = cv2.remap(frame1, map1_1, map1_2, cv2.INTER_LINEAR)
        persp1 = cv2.warpPerspective(undist1, persp_mtx1, size1)

        # 处理第二个视频帧
        undist2 = cv2.remap(frame2, map2_1, map2_2, cv2.INTER_LINEAR)
        persp2 = cv2.warpPerspective(undist2, persp_mtx2, size2)

        # 调整尺寸确保一致
        persp1 = cv2.resize(persp1, size1)
        persp2 = cv2.resize(persp2, size2)

        # 横向拼接
        combined = np.hstack((persp1, persp2))

        # 写入输出视频
        out.write(combined)

        # 显示处理进度
        preview = cv2.resize(combined, (1280, 360))
        cv2.imshow('Processing Preview', preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 处理完成！输出视频: {output_path}")


# ===== 使用示例 =====
if __name__ == "__main__":
    # 定义相机1的参数
    cam_mtx1 = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
    dist1 = np.array([-0.18, 0.02, 0, 0, 0], dtype=np.float32)
    src_pts1 = np.float32([[30, 450], [30, 30], [610, 30], [610, 450]])
    dst_pts1 = np.float32([[0, 480], [0, 0], [640, 0], [640, 480]])

    # 定义相机2的参数（可以与相机1不同）
    cam_mtx2 = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
    dist2 = np.array([-0.18, 0.02, 0, 0, 0], dtype=np.float32)
    src_pts2 = np.float32([[30, 450], [30, 30], [610, 30], [610, 450]])
    dst_pts2 = np.float32([[0, 480], [0, 0], [640, 0], [640, 480]])

    # 调用处理函数
    process_and_combine_videos(
        "camera1.mp4",
        "camera2.mp4",
        "combined_output.mp4",
        camera_params1=(cam_mtx1, dist1, src_pts1, dst_pts1),
        camera_params2=(cam_mtx2, dist2, src_pts2, dst_pts2),
        size1=(640, 480),  # 第一个视频输出尺寸
        size2=(640, 480)  # 第二个视频输出尺寸
    )