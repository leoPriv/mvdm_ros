import cv2
import numpy as np

def correct_perspective(image_path, src_points, output_size=(800, 600)):
    """
    对给定的图像进行透视变换校正，生成鸟瞰图。

    参数:
    - image_path (str): 输入图像的文件路径。
    - src_points (np.array): 包含4个源点坐标的NumPy数组，
                              顺序为 [左上, 右上, 左下, 右下]。
                              这是校正前图像中一个四边形的四个角点。
    - output_size (tuple): (宽度, 高度)，校正后输出图像的尺寸。

    返回:
    - warped_image (np.array): 校正后的鸟瞰图。
    """
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法找到或打开图像: {image_path}")

    img_h, img_w = img.shape[:2]
    out_w, out_h = output_size

    # 2. 定义目标点（一个完美的矩形）
    # 源点的顺序必须与目标点的顺序一一对应
    dst_points = np.float32([
        [0, 0],         # 对应源点的左上
        [out_w, 0],     # 对应源点的右上
        [0, out_h],     # 对应源点的左下
        [out_w, out_h]  # 对应源点的右下
    ])

    # 3. 计算透视变换矩阵
    # src_points 是 np.float32 类型的数组
    perspective_matrix = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points)

    # 4. 应用透视变换
    warped_image = cv2.warpPerspective(img, perspective_matrix, output_size)

    return warped_image

def stitch_images(img_left, img_right, blend=True):
    """
    使用特征点匹配拼接两张图像。

    参数:
    - img_left (np.array): 左侧图像。
    - img_right (np.array): 右侧图像。
    - blend (bool): 是否在重叠区域进行平滑过渡。

    返回:
    - stitched_image (np.array): 拼接后的全景图像。
    """
    # 1. 初始化ORB特征点检测器
    orb = cv2.ORB_create(nfeatures=2000)

    # 2. 在两张图中寻找关键点和描述符
    kp_left, des_left = orb.detectAndCompute(img_left, None)
    kp_right, des_right = orb.detectAndCompute(img_right, None)

    # 3. 使用暴力匹配器进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_right, des_left, k=2)

    # 4. 应用Lowe's比率测试，筛选出好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"找到 {len(good_matches)} 个高质量匹配点。")

    if len(good_matches) < 10:
        print("错误：没有找到足够的匹配点来进行拼接。")
        return None

    # 5. 提取匹配点的位置并计算单应性矩阵 (Homography)
    src_pts = np.float32([kp_right[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_left[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 6. 拼接图像
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]

    # 将右图根据H矩阵变换到左图的坐标系下
    # 画布宽度设置为两者之和，以确保内容完全可见
    result = cv2.warpPerspective(img_right, H, (w_left + w_right, h_left))

    # 将左图覆盖到画布的左侧
    if blend:
        # 创建一个简单的线性渐变蒙版
        mask = np.zeros((h_left, w_left, 3), dtype=np.float32)
        # 在可能的重叠区域（例如，图像右侧的200个像素）创建渐变
        overlap_width = 200
        for i in range(w_left):
            if i > w_left - overlap_width:
                alpha = (w_left - i) / overlap_width
                mask[:, i] = (alpha, alpha, alpha)
            else:
                mask[:, i] = (1, 1, 1)

        # 在result图像中，左图应该出现的位置
        roi = result[0:h_left, 0:w_left]

        # 混合
        blended_roi = cv2.multiply(img_left.astype(np.float32), mask)
        blended_result_roi = cv2.multiply(roi.astype(np.float32), 1 - mask)

        result[0:h_left, 0:w_left] = cv2.add(blended_roi, blended_result_roi).astype(np.uint8)

    else: # 不进行融合，直接覆盖
        result[0:h_left, 0:w_left] = img_left

    # 裁剪掉右侧多余的黑色区域
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        result = result[y:y+h, x:x+w]

    return result


if __name__ == '__main__':
    # --- 任务1: 畸变校正 ---
    # !! 重要提示 !!
    # 下面的 'src_points' 坐标是我根据您提供的图片估算的。
    # 为了达到最佳效果，您应该使用图像编辑软件（如GIMP）打开图片，
    # 精确找到车道线四边形的四个角点（左上、右上、左下、右下）的 (x, y) 像素坐标。

    # 为左侧摄像头 'view-left.png' 定义源点
    # 图像尺寸约为 691x456
    src_points_left = np.array([
        [1, 51],      # 左上角 (在最上方车道线上)
        [640, 62],     # 右上角
        [1, 371],      # 左下角 (在最下方车道线上)
        [640, 395]      # 右下角
    ])

    # 为右侧摄像头 'view-right.jpg' 定义源点
    # 图像尺寸约为 674x540
    src_points_right = np.array([
        [1, 27],      # 左上角
        [562, 16],     # 右上角
        [1, 350],      # 左下角
        [552, 358]      # 右下角
    ])

    # 定义校正后鸟瞰图的统一尺寸
    output_dimensions = (1000, 750)

    try:
        # 校正左侧图像
        corrected_left = correct_perspective(
            'view-left.png',
            src_points_left,
            output_dimensions
        )
        # 校正右侧图像
        corrected_right = correct_perspective(
            'view-right.png',
            src_points_right,
            output_dimensions
        )

        cv2.imshow('Corrected Left View', corrected_left)
        cv2.imshow('Corrected Right View', corrected_right)
        print("已生成校正后的鸟瞰图。按任意键继续进行拼接...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # --- 任务2: 图像拼接 ---
        print("\n正在拼接图像...")
        final_panorama = stitch_images(corrected_left, corrected_right, blend=True)

        if final_panorama is not None:
            # 调整最终图像大小以便于显示
            h, w = final_panorama.shape[:2]
            display_w = 1200
            display_h = int(display_w * h / w)
            final_display = cv2.resize(final_panorama, (display_w, display_h))

            cv2.imshow('Final Stitched Panorama', final_display)
            cv2.imwrite('final_panorama.jpg', final_panorama)
            print("\n任务完成！拼接后的全景图已显示，并保存为 'final_panorama.jpg'。")
            print("按任意键退出。")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"发生未知错误: {e}")