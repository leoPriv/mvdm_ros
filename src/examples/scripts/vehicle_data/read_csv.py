import pandas as pd
import glob

# 假设所有CSV文件在同一目录下，命名格式如 "123.csv"（车辆ID为123）
csv_files = glob.glob("*.csv")  # 获取所有CSV文件

vehicle_data = {}  # 最终存储的字典

for file in csv_files:
    # 从文件名提取车辆ID（去掉 ".csv" 后缀）
    vehicle_id = file.split(".")[0]

    # 读取CSV文件
    df = pd.read_csv(file)

    # 提取 velocity 和 lane_change_action 列（假设列名准确）
    velocity = df["velocity"].tolist()  # 转为列表
    lane_change_action = df["lane_change_action"].tolist()

    # 存入字典
    vehicle_data[vehicle_id] = {
        "velocity": velocity,
        "lane_change_action": lane_change_action
    }

# 打印结果示例
print(vehicle_data)