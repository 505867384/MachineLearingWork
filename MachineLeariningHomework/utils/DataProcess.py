import pandas as pd
import argparse
import os
import logging

# --- 0.参数和日志 ---
parser = argparse.ArgumentParser(description='处理原始数据')
parser.add_argument("--file_dir", type=str, default="./Original_Data", help="原始数据目录路径")
parser.add_argument("--output_dir", type=str, default="../data", help="处理后数据目录路径")
parser.add_argument("--log_path", type=str, default="../logs/数据处理日志.log", help="日志文件路径")

args = parser.parse_args()

# 创建日志和输出目录
log_dir = os.path.dirname(args.log_path)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="日志等级:%(levelname)s || 时间:%(asctime)s || 消息:%(message)s",
    handlers=[
        logging.FileHandler(args.log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# --- 1. 准备【完整】的列名 ---
full_column_names = [
    'timestamp',
    'global_active_power', 'global_reactive_power', 'voltage', 'global_intensity',
    'sub_metering_1', 'sub_metering_2', 'sub_metering_3',
    'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
]

file_list = [f for f in os.listdir(args.file_dir) if f.endswith(('.csv', '.txt'))]
logging.info(f"待处理文件列表: {file_list}")

for file_name in file_list:
    file_path = os.path.join(args.file_dir, file_name)
    logging.info(f"========== 开始处理文件: {file_path} ==========")
    
    # --- 2. 数据加载 ---
    data_all = pd.read_csv(file_path, header=None, names=full_column_names, sep=',', low_memory=False)
    logging.info(f"文件加载成功，共 {len(data_all)} 行数据。")

    # --- 3. 数据清洗 ---
    columns_to_drop = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
    data_all.drop(columns=columns_to_drop, inplace=True)
    logging.info(f"已删除天气相关列。")

    mask_has_question_mark = (data_all == '?').any(axis=1)
    if mask_has_question_mark.any():
        rows_to_drop = mask_has_question_mark.sum()
        data_all = data_all[~mask_has_question_mark].copy()
        logging.info(f"删除了 {rows_to_drop} 行包含 '?' 的数据。剩余 {len(data_all)} 行。")

    data_all['timestamp'] = pd.to_datetime(data_all['timestamp'], errors='coerce', format="%Y-%m-%d %H:%M:%S")
    rows_before_dropna = len(data_all)
    data_all.dropna(subset=['timestamp'], inplace=True)
    rows_after_dropna = len(data_all)
    if rows_before_dropna > rows_after_dropna:
        logging.info(f"删除了 {rows_before_dropna - rows_after_dropna} 行无效时间戳的数据。剩余 {rows_after_dropna} 行。")

    columns_to_process = [
        'global_active_power', 'global_reactive_power', 'voltage', 'global_intensity',
        'sub_metering_1', 'sub_metering_2', 'sub_metering_3'
    ]
    for col in columns_to_process:
        data_all[col] = pd.to_numeric(data_all[col], errors='coerce')
    data_all.fillna(0, inplace=True)
    logging.info("已将数值列转换并填充空值。")

    # --- 4. 设置索引并聚合 ---
    if data_all.empty:
        logging.warning("警告：经过清洗后，没有任何数据剩余，无法进行聚合。将跳过此文件。")
        continue

    data_all.set_index('timestamp', inplace=True)
    
    aggregation_rules = {
        'global_active_power': 'sum', 'global_reactive_power': 'sum',
        'voltage': 'mean', 'global_intensity': 'mean',
        'sub_metering_1': 'sum', 'sub_metering_2': 'sum', 'sub_metering_3': 'sum'
    }
    daily_data_all = data_all.resample('D').agg(aggregation_rules)
    daily_data_all.fillna(0, inplace=True)
    logging.info(f"数据已按天聚合，生成 {len(daily_data_all)} 条日度数据。")

    rows_before_filter = len(daily_data_all)
    daily_data_all = daily_data_all[daily_data_all['global_active_power'] > 0].copy()
    rows_after_filter = len(daily_data_all)
    if rows_before_filter > rows_after_filter:
        logging.info(f"已删除 {rows_before_filter - rows_after_filter} 条总功耗为0的日度数据。剩余 {rows_after_filter} 条。")

    # --- 5. 新增特征 ---
    
    # --- 新增：根据公式计算 sub_metering_remainder ---
    daily_data_all['sub_metering_remainder'] = \
        (daily_data_all['global_active_power'] * 1000 / 60) - \
        (daily_data_all['sub_metering_1'] + daily_data_all['sub_metering_2'] + daily_data_all['sub_metering_3'])
    logging.info("已根据公式计算并新增 sub_metering_remainder 列。")
    # --- 计算结束 ---

    daily_data_all['month'] = daily_data_all.index.month
    daily_data_all['weekday'] = daily_data_all.index.weekday + 1

    # --- 6. 格式化并保存结果 ---
    daily_data_all = daily_data_all.round(3)
    logging.info("已将所有浮点数四舍五入至三位小数。")

    output_path = os.path.join(args.output_dir, file_name)
    daily_data_all.to_csv(output_path, index=False)
    logging.info(f"处理完成！结果已成功保存到文件: {output_path}\n")