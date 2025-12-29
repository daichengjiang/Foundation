import pandas as pd
import os
import shutil
import argparse

def process_teachers(base_dir):
    # 1. 检查路径是否存在
    if not os.path.isdir(base_dir):
        print(f"错误: 找不到文件夹 '{base_dir}'")
        return

    csv_path = os.path.join(base_dir, "teacher_dynamics.csv")
    if not os.path.exists(csv_path):
        print(f"错误: 在 {base_dir} 中找不到 teacher_dynamics.csv")
        return

    # 2. 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 3. 筛选需要删除的行 (任意一个绝对值 > 0.2)
    condition = (df['x_off_mean'].abs() > 0.3) | \
                (df['y_off_mean'].abs() > 0.3) | \
                (df['z_off_mean'].abs() > 0.3)

    ids_to_delete = df[condition]['id'].tolist()
    df_keep = df[~condition].copy().sort_values('id').reset_index(drop=True)

    print(f"目标目录: {base_dir}")
    print(f"读取数据: {len(df)} 条，需删除: {len(ids_to_delete)} 条，保留: {len(df_keep)} 条")

    # 4. 删除异常 teacher 对应的文件夹
    for old_id in ids_to_delete:
        folder_name = f"teacher_{old_id:04d}"
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"已物理删除: {folder_name}")

    # 5. 重命名剩余文件夹并更新 ID 逻辑
    new_rows = []
    for new_id, row in df_keep.iterrows():
        old_id = int(row['id'])
        old_folder_name = f"teacher_{old_id:04d}"
        new_folder_name = f"teacher_{new_id:04d}"
        
        old_path = os.path.join(base_dir, old_folder_name)
        new_path = os.path.join(base_dir, new_folder_name)

        # 执行重命名 (只有当新旧路径不同时才操作)
        if old_path != new_path:
            if os.path.exists(old_path):
                # 如果目标文件夹已存在(极端情况)，先排除，正常逻辑下由于已删除坏数据，此处应通畅
                os.rename(old_path, new_path)
                print(f"重命名: {old_folder_name} -> {new_folder_name}")
        
        # 更新该行的 ID 信息
        row['id'] = new_id
        new_rows.append(row)

    # 6. 保存更新后的 CSV
    df_final = pd.DataFrame(new_rows)
    cols = ['id'] + [c for c in df_final.columns if c != 'id']
    df_final[cols].to_csv(csv_path, index=False)
    print(f"\n处理完毕！新的 CSV 已保存至: {csv_path}")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="根据动力学参数筛选并重排 teacher 文件夹及 CSV 数据")
    parser.add_argument("path", type=str, help="包含 teacher 文件夹和 csv 的目标目录路径")
    
    args = parser.parse_args()
    
    # 运行主逻辑
    process_teachers(args.path)