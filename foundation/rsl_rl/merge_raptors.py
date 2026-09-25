import os
import shutil
from pathlib import Path
import pandas as pd


def merge_raptor_folders(source_dirs, output_dir):
  """合并多个 raptor 文件夹，自动重命名 teacher_xxxx 并更新 teacher_dynamics.csv 中的 id。"""
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  all_csv_data = []
  global_new_id = 0  # 全局连续序号计数器

  # 用于最后汇总展示来源映射关系
  summary_records = []

  print('=' * 60)
  print(f'🚀 开始合并任务，目标文件夹: {output_path.resolve()}')
  print('=' * 60)

  for src_dir in source_dirs:
    src_path = Path(src_dir)
    if not src_path.exists():
      print(f'⚠️ 警告: 源路径 {src_path.resolve()} 不存在，已跳过。')
      continue

    print(f'\n📂 正在扫描源文件夹: {src_path.name}')

    # 读取当前 raptor 文件夹下的 teacher_dynamics.csv
    csv_path = src_path / 'teacher_dynamics.csv'
    df = None
    if csv_path.exists():
      df = pd.read_csv(csv_path)
      print(f'   - 成功加载 CSV: {csv_path.name} (共 {len(df)} 行记录)')
    else:
      print(f'   - ⚠️ 提示: 在 {src_path.name} 下未找到 teacher_dynamics.csv')

    # 查找并按名称排序当前目录下的所有 teacher_xxxx 文件夹
    teacher_dirs = sorted(
        [d for d in src_path.iterdir() if d.is_dir() and d.name.startswith('teacher_')]
    )
    print(f'   - 发现 {len(teacher_dirs)} 个 teacher 文件夹')

    for old_teacher_dir in teacher_dirs:
      # 生成新的文件夹名称（保持4位数补零，如 teacher_0000, teacher_0010 等）
      new_teacher_name = f'teacher_{global_new_id:04d}'
      new_teacher_path = output_path / new_teacher_name

      # 复制文件夹内容到目标路径
      if new_teacher_path.exists():
        shutil.rmtree(new_teacher_path)
      shutil.copytree(old_teacher_dir, new_teacher_path)

      # 尝试解析旧文件夹中的数字 ID 并更新 CSV 对应行
      matched_csv = False
      try:
        old_id_str = old_teacher_dir.name.split('_')[1]
        old_id = int(old_id_str)

        if df is not None and 'id' in df.columns:
          row_mask = df['id'] == old_id
          if row_mask.any():
            row_data = df[row_mask].copy()
            row_data['id'] = global_new_id
            all_csv_data.append(row_data)
            matched_csv = True
      except Exception as e:
        print(f'     -> 解析 {old_teacher_dir.name} 的 ID 时出错: {e}')

      # 记录操作日志
      print(
          f'   [复制] {src_path.name}/{old_teacher_dir.name}'
          f'  --->  {output_path.name}/{new_teacher_name}'
          f" {'(CSV已关联更新)' if matched_csv else '(未找到对应CSV行)'}"
      )

      # 收集来源记录，用于最后汇总
      summary_records.append({
          'new_name': new_teacher_name,
          'source_folder': src_path.name,
          'old_name': old_teacher_dir.name,
          'new_id': global_new_id,
      })

      global_new_id += 1

  # 合并所有 CSV 数据并写入目标文件夹
  if all_csv_data:
    merged_df = pd.concat(all_csv_data, ignore_index=True)
    merged_csv_path = output_path / 'teacher_dynamics.csv'
    merged_df.to_csv(merged_csv_path, index=False)
    print('\n' + '=' * 60)
    print('✨ 合并成功完成！')
    print('=' * 60)
    print(f'📁 目标总文件夹: {output_path.resolve()}')
    print(f'📊 总计合并了 {global_new_id} 个 teacher 文件夹')
    print(f'📝 合并后的 CSV 路径: {merged_csv_path}')

    # 打印详细的来源映射汇总表
    print('\n📋 【新生成的 raptor 文件夹内容及来源对照表】:')
    print(f'{"新文件夹名称":<15} | {"来源文件夹":<15} | {"原文件夹名称":<15}')
    print('-' * 53)
    for record in summary_records:
      print(
          f"{record['new_name']:<15} |"
          f" {record['source_folder']:<15} |"
          f" {record['old_name']:<15}"
      )
    print('=' * 60)
  else:
    print('\n⚠️ 警告: 没有收集到任何有效的 CSV 数据。')


if __name__ == '__main__':
  # 方案 A：使用相对路径（从 foundation/rsl_rl/ 往上退两层，再进 logs/...）
  base_dir = Path(__file__).resolve().parent.parent.parent / 'logs/rsl_rl/c5_teachers'

  source_folders = [
      base_dir / 'raptor0',
      base_dir / 'raptor1',
      base_dir / 'raptor2',
      base_dir / 'raptor3',
      base_dir / 'raptor4',
      base_dir / 'raptor5',
      base_dir / 'raptor6',
      base_dir / 'raptor7',
      base_dir / 'raptor8',
      base_dir / 'raptor9',
      base_dir / 'raptor10',
      base_dir / 'raptor11',
      base_dir / 'raptor12',
      base_dir / 'raptor13',
      base_dir / 'raptor14',
      base_dir / 'raptor15',
      base_dir / 'raptor16',
      base_dir / 'raptor17',
      base_dir / 'raptor18',
      base_dir / 'raptor19',
      base_dir / 'raptor20',
      base_dir / 'raptor21',
      base_dir / 'raptor22',
      base_dir / 'raptor23',
      base_dir / 'raptor24',
      base_dir / 'raptor25',
      base_dir / 'raptor26',
      base_dir / 'raptor27',
      base_dir / 'raptor28',
      base_dir / 'raptor29',
      base_dir / 'raptor30',
      base_dir / 'raptor31',
      base_dir / 'raptor32',
      base_dir / 'raptor33',
      base_dir / 'raptor34',
      base_dir / 'raptor35',
  ]

  # 输出合并后的文件夹（比如依然放在 multi_teachers/raptor 下）
  target_folder = base_dir / 'raptor'

  merge_raptor_folders(source_folders, target_folder)