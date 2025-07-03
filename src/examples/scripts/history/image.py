import cv2
import numpy as np


class LaneKeepingSystem:
    def __init__(self):
        self.last_steering = 0
        self.lane_width = 300  # 车道宽度（像素）

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def region_of_interest(self, img):
        height, width = img.shape
        mask = np.zeros_like(img)

        polygon = np.array([[
            (width * 0.1, height),
            (width * 0.45, height * 0.6),
            (width * 0.55, height * 0.6),
            (width * 0.9, height),
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    def detect_lane_lines(self, img):
        lines = cv2.HoughLinesP(img, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=100)
        return lines

    def calculate_lane_center(self, lines, height):
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 0.0001)
            if slope < -0.5:
                left_lines.append(line[0])
            elif slope > 0.5:
                right_lines.append(line[0])

        def get_average_line(lines):
            if not lines:
                return None
            return np.mean(lines, axis=0)

        avg_left = get_average_line(left_lines)
        avg_right = get_average_line(right_lines)

        if avg_left is not None and avg_right is not None:
            lx1, ly1, lx2, ly2 = avg_left
            rx1, ry1, rx2, ry2 = avg_right

            left_x = lx1 + (height - ly1) * (lx2 - lx1) / (ly2 - ly1 + 1e-5)
            right_x = rx1 + (height - ry1) * (rx2 - rx1) / (ry2 - ry1 + 1e-5)
            lane_center_x = int((left_x + right_x) / 2)
            return lane_center_x
        elif avg_left is not None:
            lx1, ly1, lx2, ly2 = avg_left
            left_x = lx1 + (height - ly1) * (lx2 - lx1) / (ly2 - ly1 + 1e-5)
            return int(left_x + self.lane_width / 2)
        elif avg_right is not None:
            rx1, ry1, rx2, ry2 = avg_right
            right_x = rx1 + (height - ry1) * (rx2 - rx1) / (ry2 - ry1 + 1e-5)
            return int(right_x - self.lane_width / 2)
        else:
            return None


def main():
    lks = LaneKeepingSystem()

    # 读取图像
    image = cv2.imread("./picture/04-中间-左转角.png")  # 替换为你的图像路径
    height, width = image.shape[:2]

    edges = lks.preprocess_image(image)
    roi = lks.region_of_interest(edges)
    lines = lks.detect_lane_lines(roi)

    # 拷贝一份图像用于绘制
    debug_img = image.copy()

    if lines is not None:
        # 绘制检测到的线条
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 计算车道中心
        lane_center_x = lks.calculate_lane_center(lines, height)
        if lane_center_x is not None:
            # 绘制车道中心线（绿色）
            cv2.line(debug_img, (lane_center_x, 0), (lane_center_x, height), (0, 255, 0), 2)

    # 绘制车辆中心线（蓝色）
    vehicle_center_x = width // 2
    cv2.line(debug_img, (vehicle_center_x, 0), (vehicle_center_x, height), (255, 0, 0), 2)

    # 显示图像（放大显示）
    scale = 2
    big_img = cv2.resize(debug_img, (width * scale, height * scale))
    cv2.imshow("Lane Detection", big_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
