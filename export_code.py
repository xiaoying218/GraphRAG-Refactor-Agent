import os
import shutil

def export_and_zip(source_dir, target_dir, ignore_dirs=None, ignore_files=None):
    """
    1. 将 source_dir 下的所有 .py 文件复制到 target_dir
    2. 将 target_dir 打包成 zip 文件
    """
    if ignore_dirs is None:
        # 默认过滤的无关目录（数据、环境、Git）
        ignore_dirs = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'wandb', 'data', 'logs', 'output']
    
    if ignore_files is None:
        ignore_files = []

    count = 0
    ignored_count = 0
    
    print(f"🚀 [第一步] 开始从 [{os.path.basename(source_dir)}] 提取代码...")

    for root, dirs, files in os.walk(source_dir):
        # 1. 过滤文件夹
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for file in files:
            # 2. 基础过滤：必须是 .py 文件
            if file.endswith('.py'):
                
                # 3. 黑名单过滤（精确匹配）
                if file in ignore_files:
                    print(f"   [🚫 忽略] {file}")
                    ignored_count += 1
                    continue
                
                # --- 复制逻辑 ---
                src_file_path = os.path.join(root, file)
                # 计算相对路径，保持原有结构
                relative_path = os.path.relpath(src_file_path, source_dir)
                dest_file_path = os.path.join(target_dir, relative_path)
                
                # 创建父目录
                os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                # 复制文件
                shutil.copy2(src_file_path, dest_file_path)
                print(f"   [✅ 复制] {relative_path}")
                count += 1

    print("-" * 30)
    print(f"📊 统计：复制了 {count} 个文件，忽略了 {ignored_count} 个文件。")

    # --- 第二步：压缩逻辑 ---
    print(f"📦 [第二步] 正在打包成 ZIP...")
    
    # base_name: 压缩包路径（不带后缀），shutil会自动添加 .zip
    shutil.make_archive(base_name=target_dir, format='zip', root_dir=target_dir)

    print(f"🎉 全部搞定！")
    print(f"   🤐 压缩包: {target_dir}.zip")

if __name__ == '__main__':
    # ================= 配置区域 =================
    
    # 你的项目源路径
    my_project_path = '/Users/lixiaoying/lxy/code/Code Refactoring/graph_rag_context_engine'
    
    # 你的输出路径
    output_path = '/Users/lixiaoying/lxy/code/Code Refactoring/backup'
    
    # 保持默认忽略的文件夹（如 .git, data 等）
    ignore_dirs_list = ['.git', '__pycache__', 'venv', 'data', 'logs', 'checkpoints', 'output', 'tmp', '.DS_Store']
    
    # 你指定的忽略文件列表
    ignore_files_list = [
        'merge_refactor_logs_copy.py', 
        'merge_refactor_logs.py', 
        'merge_v2.py', 
        'merge.py',
        'export_code.py',
        'test.py'
    ]

    # ===========================================
    
    export_and_zip(my_project_path, output_path, ignore_dirs_list, ignore_files_list)