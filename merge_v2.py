import os
import sys
import argparse

def merge_project_code(root_dir, output_file, target_ext):
    # 1. 【黑名单目录名】遇到这些文件夹名，直接跳过
    IGNORE_DIRS = {
        # Python 相关
        '.venv', 'venv', 'env', '.env', 
        '__pycache__', 'site-packages', 'egg-info',
        # Java/Maven/Gradle/IDEA 相关 (新增)
        'target', 'build', 'out', 'bin', 
        '.mvn', 'gradle', '.gradle',
        # 通用/编辑器配置
        '.git', '.idea', '.vscode', '.settings',
        'node_modules', 'dist'
    }

    # 2. 【路径关键词过滤】路径里包含这些词强制跳过
    IGNORE_KEYWORDS = [
        'site-packages',
        os.sep + 'lib' + os.sep + 'python',  # 跨平台匹配 /lib/python
        'test-classes'  # Java 测试编译目录
    ]

    this_script_name = os.path.basename(__file__)
    file_count = 0
    
    # 获取绝对路径，显示更清晰
    abs_root = os.path.abspath(root_dir)
    
    print(f"正在扫描目录: {abs_root}")
    print(f"目标文件类型: {target_ext}")
    print("已开启【强力过滤模式】(屏蔽虚拟环境、target、node_modules等)...")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(root_dir):
            # --- 核心过滤步骤 1: 修改 dirs 列表 (阻止进入黑名单目录) ---
            # 倒序遍历以安全移除元素
            for i in range(len(dirs) - 1, -1, -1):
                d = dirs[i]
                # 规则：在黑名单中，或以 '.' 开头（隐藏目录，如 .git, .idea）
                if d in IGNORE_DIRS or d.startswith('.'):
                    dirs.pop(i)
            
            # --- 核心过滤步骤 2: 路径关键词检查 ---
            if any(keyword in root for keyword in IGNORE_KEYWORDS):
                continue

            for file in files:
                # 核心判断：后缀名匹配 且 不是脚本自己
                if file.endswith(target_ext) and file != this_script_name:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, root_dir)
                    
                    print(f"正在合并: {relative_path}")
                    
                    outfile.write(f"\n{'='*20} FILE: {relative_path} {'='*20}\n")
                    
                    try:
                        with open(full_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except UnicodeDecodeError:
                        # Java有时候会有GBK编码，尝试容错，或者直接跳过
                        outfile.write(f"# 读取错误: 编码不是 UTF-8，已跳过内容\n")
                    except Exception as e:
                        outfile.write(f"# 读取错误: {e}\n")
                        
                    outfile.write(f"\n{'='*20} END FILE {'='*20}\n\n")
                    file_count += 1

    print(f"\n合并完成！共处理 {file_count} 个文件。")
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    # 配置命令行参数解析
    parser = argparse.ArgumentParser(description="合并指定目录下的代码文件 (Python 或 Java)")
    
    # 添加位置参数 'lang'，限定只能选 python 或 java
    parser.add_argument('lang', choices=['python', 'java'], 
                        help="指定要合并的语言: 输入 'python' 或 'java'")
    
    # 解析参数
    # 如果用户没输参数，argparse 会自动提示错误并显示 help
    args = parser.parse_args()

    # 映射配置
    CONFIG = {
        'python': {'ext': '.py', 'out': 'final_code_python.txt'},
        'java':   {'ext': '.java', 'out': 'final_code_java.txt'}
    }

    current_config = CONFIG[args.lang]
    
    # 执行合并
    # os.getcwd() 获取当前运行脚本的目录
    merge_project_code(
        root_dir=os.getcwd(), 
        output_file=current_config['out'], 
        target_ext=current_config['ext']
    )