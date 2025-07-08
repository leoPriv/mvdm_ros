import pickle

# 从 .pkl 文件加载数据
with open("log_data.pkl", "rb") as f:  # 必须用二进制模式 "rb"
    log_data = pickle.load(f)



print('da')