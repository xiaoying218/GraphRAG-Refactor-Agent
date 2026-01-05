import os

def merge_project_code(root_dir, output_file):
    # 1. 【黑名单目录名】遇到这些文件夹名，直接跳过，连进都不进去
    # 这些通常是虚拟环境、缓存、Git配置等
    IGNORE_DIRS = {
        '.venv', 'venv', 'env', '.env',  # 各种虚拟环境写法
        '__pycache__', 
        '.git', '.idea', '.vscode',      # 编辑器配置
        'build', 'dist', 'egg-info',     # 打包残留
        'node_modules',                  # 前端依赖（如果有）
        'site-packages'                  # 严防死守第三方库
    }

    # 2. 【路径关键词过滤】如果文件路径里包含这些词，也强制跳过
    # 这是为了防止漏网之鱼，比如某些深层目录里的库文件
    IGNORE_KEYWORDS = [
        'site-packages',  # 再次确保库文件不被读取
        '/lib/python',    # 典型的库路径特征
        '\\lib\\python'   # 兼容 Windows 路径
    ]

    this_script_name = os.path.basename(__file__)
    file_count = 0
    
    print(f"正在扫描: {root_dir}")
    print("已开启【强力过滤模式】，自动屏蔽虚拟环境和第三方库...")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(root_dir):
            # --- 核心过滤步骤 1: 修改 dirs 列表，阻止进入黑名单目录 ---
            # 这一步非常关键，它能让程序直接不扫描 .venv 文件夹，节省大量时间
            # 我们倒序遍历，这样移除元素不会影响索引
            for i in range(len(dirs) - 1, -1, -1):
                d = dirs[i]
                # 规则：如果目录名在黑名单里，或者以 '.' 开头（隐藏目录），直接剔除
                if d in IGNORE_DIRS or d.startswith('.'):
                    dirs.pop(i)
            
            # --- 核心过滤步骤 2: 检查当前路径是否包含敏感词 ---
            # 如果路径里居然还有 site-packages (双重保险)，直接跳过这一层的所有文件
            if any(keyword in root for keyword in IGNORE_KEYWORDS):
                continue

            for file in files:
                if file.endswith('.py') and file != this_script_name:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, root_dir)
                    
                    # 打印一下，让你知道它在干活，但没在瞎干活
                    print(f"正在合并: {relative_path}")
                    
                    outfile.write(f"\n{'='*20} FILE: {relative_path} {'='*20}\n")
                    
                    try:
                        with open(full_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"# 读取错误: {e}\n")
                        
                    outfile.write(f"\n{'='*20} END FILE {'='*20}\n\n")
                    file_count += 1

    print(f"\n合并完成！共处理 {file_count} 个文件。")
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    merge_project_code(os.getcwd(), "final_code.txt")
