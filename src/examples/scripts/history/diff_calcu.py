import cv2
import pandas as pd
import numpy as np
from shapely.geometry import LineString
import os

save_folder = './0605172352'
input_folder = save_folder + '/raw_picture/'            # 原图文件夹
output_folder = save_folder + '/results/'          # 处理后图像输出文件夹
os.makedirs(output_folder, exist_ok=True)
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
def get_angle(line):
    x1, y1, x2, y2 = line[0]
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

def merge_lines_by_angle(lines, angle_threshold=10):
    """
    按角度聚类，使用 cv2.fitLine 拟合每一组直线
    """
    if lines is None:
        return []

    angle_groups = []
    for line in lines:
        angle = get_angle(line)
        # print(angle)
        if abs(angle) != 0 and abs(angle) != 90:
            matched = False
            for group in angle_groups:
                if abs(get_angle(group[0]) - angle) < angle_threshold:
                    group.append(line)
                    matched = True
                    break
            if not matched:
                angle_groups.append([line])

    merged_lines = []
    for group in angle_groups:
        points = []
        for line in group:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(slope) > 0.5:
                points.append([x1, y1])
                points.append([x2, y2])
        points = np.array(points)
        if len(points) >= 2:
            [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            # 延展线段
            x1 = int(x0.item() - vx.item() * 1000)
            y1 = int(y0.item() - vy.item() * 1000)
            x2 = int(x0.item() + vx.item() * 1000)
            y2 = int(y0.item() + vy.item() * 1000)
            merged_lines.append(((x1, y1), (x2, y2)))
    return merged_lines
index = 0
diff = []
for file_name in image_files:
    image_path = os.path.join(input_folder, file_name)
    img = cv2.imread(image_path)

    height, width = img.shape[:2]
    lower_half = img[round(height // 1.75): height, round(width // 1.75): width]
    line_img = lower_half.copy()
    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    # 可视化边缘检测结果
    # cv2.imshow('Canny Edges', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)

    merged = merge_lines_by_angle(lines)
    if len(merged) > 0 and len(merged[0]) > 1:
        left_k = (merged[0][1][1] - merged[0][0][1]) / (merged[0][1][0] - merged[0][0][0] + 1e-6)
    else:
        left_k = None
    if len(merged) > 1 and len(merged[1]) > 1:
        right_k = (merged[1][1][1] - merged[1][0][1]) / (merged[1][1][0] - merged[1][0][0] + 1e-6)
    else:
        right_k = None



    # print(left_k, right_k)
    # 画合并后的长线
    inter_points = []
    height_half, width_half = line_img.shape[:2]
    line0 = LineString([(-100, height_half-1), (width_half+100, height_half-1)])
    for pt1, pt2 in merged:
    #     line1 = LineString([pt1, pt2])
    #     inter_p = line1.intersection(line0)
    #     inter_points.append((int(inter_p.x),int(inter_p.y)))
    #     cv2.circle(line_img, (int(inter_p.x),int(inter_p.y)), 10, (255, 0, 0), -1)
        cv2.line(line_img, pt1, pt2, (0, 0, 255), 3)
    # centre_point_x = (inter_points[1][0]+inter_points[0][0])//2
    # centre_point_y = (inter_points[1][1]+inter_points[0][1])//2
    # cv2.circle(line_img, (centre_point_x, centre_point_y), 10, (0, 0, 255), -1)
    # if index == 0:
    #     centre_point_0 = centre_point_x
    # diff_0 = centre_point_x - centre_point_0
    # print(diff_0)
    # diff.append({'Iteration': index, 'Difference': diff_0})
    index += 1

    # 保存结果图
    output_path = os.path.join(output_folder, f'merged_{file_name}')
    cv2.imwrite(output_path, line_img)
    # print(f'Processed: {file_name}')
pd.DataFrame(diff).to_csv(save_folder+'/csv/diff.csv', index=False)







