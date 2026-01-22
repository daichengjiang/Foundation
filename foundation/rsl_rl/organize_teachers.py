import os
import csv
import argparse

def parse_range_pair(pair_str):
    """
    解析 '500-598:0-98' 格式的字符串
    """
    try:
        old_part, new_part = pair_str.split(':')
        def get_range(s):
            start, end = map(int, s.split('-'))
            return list(range(start, end + 1))
        
        old_ids = get_range(old_part)
        new_ids = get_range(new_part)
        
        if len(old_ids) != len(new_ids):
            raise ValueError(f"Range lengths do not match! {len(old_ids)} vs {len(new_ids)}")
        return list(zip(old_ids, new_ids))
    except Exception as e:
        print(f"Error parsing range pair '{pair_str}': {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Rename teacher folders and reorder CSV with precision preservation.")
    parser.add_argument("--timestamp", type=str, required=True, help="The timestamp folder name")
    parser.add_argument("--mapping", type=str, required=True, 
                        help="Example: '500-510:0-10,800-805:20-25'")
    parser.add_argument("--log_root", type=str, default="logs/rsl_rl/multi_teachers", help="Root path")
    
    args = parser.parse_args()

    base_dir = os.path.join(args.log_root, args.timestamp)
    csv_path = os.path.join(base_dir, "teacher_dynamics.csv")

    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return

    # 1. 解析映射
    mapping_pairs = []
    for segment in args.mapping.split(','):
        mapping_pairs.extend(parse_range_pair(segment.strip()))
    
    id_map = {str(old_id): str(new_id) for old_id, new_id in mapping_pairs}
    if not id_map:
        print("No valid mappings found.")
        return

    # 2. 重命名文件夹 (逻辑保持不变)
    temp_renames = []
    for old_id_str, new_id_str in id_map.items():
        # 这里需要补零到4位用于匹配文件夹
        old_folder = os.path.join(base_dir, f"teacher_{int(old_id_str):04d}")
        if os.path.exists(old_folder):
            temp_name = os.path.join(base_dir, f"temp_teacher_{old_id_str}_{new_id_str}")
            os.rename(old_folder, temp_name)
            temp_renames.append((temp_name, int(new_id_str)))
        else:
            print(f"Warning: Folder {old_folder} not found, skipping.")

    for temp_path, new_id_int in temp_renames:
        new_folder = os.path.join(base_dir, f"teacher_{new_id_int:04d}")
        os.rename(temp_path, new_folder)
        print(f"Renamed Folder: .../teacher_{new_id_int:04d}")

    # 3. 处理 CSV (使用 csv 模块以保持原始字符串)
    if os.path.exists(csv_path):
        rows = []
        header = None
        
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            for row in reader:
                old_id = row['id']
                if old_id in id_map:
                    row['id'] = id_map[old_id]
                rows.append(row)
        
        # 按 ID 数值大小进行排序
        # 注意：这里 row['id'] 是字符串，排序时要转成 int
        rows.sort(key=lambda x: int(x['id']))

        # 写回文件
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        
        print(f"CSV processed and sorted. Precision preserved (treated as strings).")
    else:
        print("Warning: teacher_dynamics.csv not found.")

if __name__ == "__main__":
    main()