import cv2
import numpy as np

def generate_target_control_points(source_points, image_shape, output_size, num_h_lines):
    """
    根据源控制点和期望的布局，自动生成目标控制点。
    假设所有点按行分组，并且我们希望将它们映射到一个均匀的网格上。
    """
    out_w, out_h = output_size
    img_w = image_shape[1]

    # 将y坐标均匀分布在垂直空间中
    # 在顶部和底部留出一些边距，所以我们将空间分成 num_h_lines + 1 份
    y_coords = np.linspace(out_h / (num_h_lines + 1), out_h * num_h_lines / (num_h_lines + 1), num_h_lines)

    target_points = []
    points_per_line = len(source_points) // num_h_lines

    for i in range(num_h_lines):
        for j in range(points_per_line):
            point_index = i * points_per_line + j
            src_x = source_points[point_index][0]

            # 按比例缩放x坐标
            dst_x = src_x * (out_w / img_w)
            dst_y = y_coords[i]
            target_points.append([dst_x, dst_y])

    return np.array(target_points, dtype=np.float32)


def correct_with_tps(image_path, source_points, output_size=(1000, 750), num_lines=3):
    """
    使用薄板样条插值(TPS)来校正镜头和透视畸变。

    参数:
    - image_path (str): 输入图像的路径。
    - source_points (np.array): N x 2 的源控制点数组。
    - output_size (tuple): 输出图像的 (宽度, 高度)。
    - num_lines (int): 场景中的车道线数量。

    返回:
    - warped_image (np.array): 校正后的鸟瞰图。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法找到或打开图像: {image_path}")

    # 1. 创建TPS变换器
    tps = cv2.createThinPlateSplineShapeTransformer()

    # 2. 生成我们期望的目标控制点（一个完美的水平网格）
    target_points = generate_target_control_points(source_points, img.shape, output_size, num_lines)

    # 3. 准备OpenCV所需格式的控制点 (N, 1, 2)
    source_points_cv = source_points.reshape(-1, 1, 2).astype(np.float32)
    target_points_cv = target_points.reshape(-1, 1, 2).astype(np.float32)

    # 4. 估算TPS变换
    matches = [cv2.DMatch(i, i, 0) for i in range(len(source_points))]
    # *** 代码修正 #1: 调整参数顺序 ***
    # 计算从 source_points (moving) 到 target_points (target) 的变换
    tps.estimateTransformation(source_points_cv, target_points_cv, matches)

    # 5. 应用变换来扭曲图像
    # *** 代码修正 #2: 修正warpImage调用方式 ***
    # a. 创建一个所需尺寸的目标画布
    height, width = output_size[1], output_size[0]
    dst_canvas = np.zeros((height, width, 3), dtype=img.dtype)

    # b. 将源图像扭曲并绘制到目标画布上
    #    该函数会修改并返回 dst_canvas
    warped_image = tps.warpImage(img, dst_canvas, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

    return warped_image

def stitch_images(img_left, img_right):
    """
    使用特征点匹配拼接两张图像。(此函数与之前版本相同)
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
    if matches:
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

    result = cv2.warpPerspective(img_right, H, (w_left + w_right, h_left))
    result[0:h_left, 0:w_left] = img_left

    # 裁剪掉右侧多余的黑色区域
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 寻找最大的轮廓
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        result = result[y:y+h, x:x+w]

    return result

if __name__ == '__main__':
    # --- 任务1: 使用TPS进行畸变校正 ---
    # !! 重要提示 !!
    # 这是最关键的一步。我通过在图像编辑器中查看，手动提取了以下控制点。
    # 每个子列表代表一条车道线，包含该线上从左到右的几个点的 [x, y] 坐标。
    # 为了获得最佳效果，您应该亲自提取这些点。点的数量和位置可以调整。

    # 为 'view-left.png' (691x456) 定义源控制点
    # 7条线, 每条线3个点 (左, 中, 右)
    src_points_left = np.array([
        # Line 1 (Top)
        [1, 51], [326, 48], [640, 62],
        # Line 4
        [1, 209], [324, 216], [640, 226],
        # Line 7 (Bottom)
        [1, 371], [388, 392], [640, 395],
    ])

    # 为 'view-right.jpg' (674x540) 定义源控制点
    src_points_right = np.array([
        # Line 1 (Top)
        [1, 27], [293, 16], [562, 16],
        # Line 4
        [1, 191], [299, 191], [562, 191],
        # Line 7 (Bottom)
        [1, 350], [337, 455], [294, 359],
    ])

    output_dimensions = (1000, 750) # 定义校正后鸟瞰图的统一尺寸


    print("正在校正左侧图像...")
    corrected_left = correct_with_tps(
        'view-left.png',
        src_points_left,
        output_dimensions,
        num_lines=3
    )
    # 校正右侧图像
    print("正在校正右侧图像...")
    corrected_right = correct_with_tps(
        'view-right.png',
        src_points_right,
        output_dimensions,
        num_lines=3
    )

    # cv2.imshow('Corrected Left View (TPS)', corrected_left)
    # cv2.imshow('Corrected Right View (TPS)', corrected_right)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # --- 任务2: 图像拼接 ---
    print("\n正在拼接图像...")
    final_panorama = stitch_images(corrected_left, corrected_right)

    if final_panorama is not None:
        # 调整最终图像大小以便于显示
        h, w = final_panorama.shape[:2]
        display_w = 1200
        display_h = int(display_w * h / w)
        final_display = cv2.resize(final_panorama, (display_w, display_h), interpolation=cv2.INTER_AREA)

        cv2.imshow('Final Stitched Panorama', final_display)
        cv2.imwrite('final_panorama_tps.jpg', final_panorama)
        print("\n任务完成！拼接后的全景图已显示，并保存为 'final_panorama_tps.jpg'。")
        print("按任意键退出。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
