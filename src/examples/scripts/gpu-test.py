import cv2
import inspect
# # 检查 CUDA 设备数量
cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
print(cv2.__version__)
print(f"Number of CUDA devices: {cuda_device_count}")
print(cv2.cuda.getCudaEnabledDeviceCount())  # 应该 > 0

# 检查 cv2.cuda 是否有 GaussianBlur
print(hasattr(cv2.cuda, "HoughLinesDetector"))
# print(cv2.getBuildInformation())
# 打印函数的参数列表
print(inspect.signature(cv2.cuda.createHoughLinesDetector))
