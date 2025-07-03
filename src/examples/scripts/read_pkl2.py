import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import os


def analyze_vehicle_data(pkl_file_path):
    """
    分析车辆轨迹数据，提取速度和换道动作信息

    数据结构：
    - 时间序列字典，每个时间步包含6个列表：
    [0] 车辆ID (flow_xxx.x)
    [1] 所在车道
    [2] 纵向位置
    [3] 第四个参数（忽略）
    [4] 第五个参数（忽略）
    [5] 纵向速度

    换道动作编码：
    -1: 向左变道
     0: 车道保持
     1: 向右变道
    """

    print(f"正在加载数据文件: {pkl_file_path}")

    # 加载数据
    with open(pkl_file_path, 'rb') as f:
        time_series_data = pickle.load(f)

    print(f"数据类型: {type(time_series_data)}")
    print(f"时间步数量: {len(time_series_data)}")

    # 存储处理后的数据
    vehicle_trajectories = defaultdict(lambda: {
        'time_steps': [],
        'lanes': [],
        'positions': [],
        'velocities': [],
        'lane_changes': []
    })

    # 遍历每个时间步
    for time_step, step_datas in enumerate(time_series_data):
        if time_step % 100 == 0:
            print(f"处理进度: {time_step}/{len(time_series_data)}")
        step_data = time_series_data[step_datas]
        if len(step_data) != 6:
            print(f"警告: 时间步 {time_step} 的数据长度不是6，而是 {len(step_data)}")
            continue

        vehicle_ids = step_data[0]  # 车辆ID列表
        lanes = step_data[1]  # 车道列表
        positions = step_data[2]  # 纵向位置列表
        velocities = step_data[5]  # 纵向速度列表

        # 处理每辆车的数据
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle_trajectories[vehicle_id]['time_steps'].append(time_step)
            vehicle_trajectories[vehicle_id]['lanes'].append(lanes[i])
            vehicle_trajectories[vehicle_id]['positions'].append(positions[i])
            vehicle_trajectories[vehicle_id]['velocities'].append(velocities[0][i])

    print("正在计算换道动作...")

    # 计算换道动作
    for vehicle_id, trajectory in vehicle_trajectories.items():
        lane_changes = []

        for j in range(len(trajectory['lanes'])):
            if j == 0:
                # 第一个时间步，假设为车道保持
                lane_changes.append(0)
            else:
                current_lane = trajectory['lanes'][j]
                previous_lane = trajectory['lanes'][j - 1]

                if current_lane > previous_lane:
                    # 向右变道
                    lane_changes.append(1)
                elif current_lane < previous_lane:
                    # 向左变道
                    lane_changes.append(-1)
                else:
                    # 车道保持
                    lane_changes.append(0)

        trajectory['lane_changes'] = lane_changes

    return dict(vehicle_trajectories)


def export_individual_csv_files(vehicle_trajectories, output_dir='vehicle_data2'):
    """为每个车辆ID单独导出CSV文件"""

    print("正在为每个车辆创建单独的CSV文件...")

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    exported_files = []

    for vehicle_id, trajectory in vehicle_trajectories.items():
        # 安全的文件名（替换可能有问题的字符）
        safe_filename = vehicle_id.replace('.', '_').replace('/', '_').replace('\\', '_')
        output_file = os.path.join(output_dir, f"{safe_filename}.csv")

        # 准备该车辆的数据
        vehicle_data = []

        for i in range(len(trajectory['time_steps'])):
            vehicle_data.append({
                'vehicle_id': vehicle_id,
                'time_step': trajectory['time_steps'][i],
                'lane': trajectory['lanes'][i],
                'position': trajectory['positions'][i],
                'velocity': trajectory['velocities'][i],
                'lane_change_action': trajectory['lane_changes'][i]
            })

        # 创建DataFrame并保存
        df = pd.DataFrame(vehicle_data)
        df.to_csv(output_file, index=False)

        exported_files.append(output_file)
        print(f"  导出 {vehicle_id}: {len(vehicle_data)} 条数据 -> {output_file}")

    return exported_files





def create_file_index(exported_files, vehicle_trajectories, index_file='file_index.txt'):
    """创建文件索引，方便查找各个车辆的数据文件"""

    print("正在创建文件索引...")

    with open(index_file, 'w', encoding='utf-8') as f:
        f.write("车辆轨迹数据文件索引\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总车辆数量: {len(exported_files)}\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n\n")

        f.write("车辆ID -> 文件路径 (数据点数量)\n")
        f.write("-" * 50 + "\n")

        # 按车辆ID排序
        sorted_items = sorted(vehicle_trajectories.items(), key=lambda x: x[0])

        for vehicle_id, trajectory in sorted_items:
            safe_filename = vehicle_id.replace('.', '_').replace('/', '_').replace('\\', '_')
            file_path = f"vehicle_data/{safe_filename}.csv"
            data_count = len(trajectory['time_steps'])
            f.write(f"{vehicle_id} -> {file_path} ({data_count} 条数据)\n")

    print(f"文件索引已创建: {index_file}")




def main():
    """主函数"""
    pkl_file_path = 't_s_store.pkl'

    try:
        # 分析数据
        print("开始分析车辆轨迹数据...")
        vehicle_data = analyze_vehicle_data(pkl_file_path)

        # 为每个车辆导出单独的CSV文件
        exported_files = export_individual_csv_files(vehicle_data)


        # 创建文件索引
        create_file_index(exported_files, vehicle_data)
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)

        return vehicle_data, exported_files

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    vehicle_data, exported_files= main()