import os
import json
from pathlib import Path

# ================= 配置区域 =================
# 你的 bench_out 输出目录名称
BENCH_OUT_DIR = "bench_out"
# 最终生成的合并文件名称
OUTPUT_FILE = "merged_refactor_debug_log.txt"
# ===========================================

def read_file_content(file_path):
    """读取文件内容，处理编码问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception:
            return "[Binary or Unreadable File]"
    except FileNotFoundError:
        return "[File Not Found]"

def write_separator(f, title, char="="):
    """写入清晰的分隔符"""
    f.write(f"\n{char*50}\n")
    f.write(f" {title}\n")
    f.write(f"{char*50}\n\n")

def main():
    root_path = Path(os.getcwd())
    bench_path = root_path / BENCH_OUT_DIR
    
    if not bench_path.exists():
        print(f"❌ 错误: 找不到目录 {bench_path}")
        return

    print(f"📂 开始扫描 {bench_path}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        outfile.write(f"Refactoring Agent Debug Log\nGenerated Time: {os.times()}\n\n")

        # 遍历 bench_out 下的所有模式 (例如 graph_rag, vector_only)
        for mode_dir in bench_path.iterdir():
            if not mode_dir.is_dir() or mode_dir.name.startswith("."):
                continue
            
            # 遍历模式下的具体任务 (例如 consolidate_base_score_computation)
            for task_dir in mode_dir.iterdir():
                if not task_dir.is_dir() or task_dir.name.startswith("."):
                    continue

                task_name = task_dir.name
                mode_name = mode_dir.name
                
                print(f"  Processing Task: [{mode_name}] {task_name}")
                
                write_separator(outfile, f"TASK: {task_name} (Mode: {mode_name})", char="#")

                # 1. 读取 Bench Out 目录下的概览文件
                bench_files = ["agent_summary.json", "run_record.json", "context_coverage.json"]
                artifacts_path = None
                
                for filename in bench_files:
                    file_path = task_dir / filename
                    if file_path.exists():
                        content = read_file_content(file_path)
                        outfile.write(f"--- [Bench File] {filename} ---\n")
                        outfile.write(content + "\n\n")
                        
                        # 如果是 summary，尝试提取 artifacts 路径
                        if filename == "agent_summary.json":
                            try:
                                data = json.loads(content)
                                if "artifacts_dir" in data:
                                    raw_path = data["artifacts_dir"]
                                    # 处理绝对路径，如果在这台机器上跑，绝对路径通常是有效的
                                    # 如果绝对路径无效，尝试将其视为相对路径或寻找项目内的对应路径
                                    artifacts_path = Path(raw_path)
                                    if not artifacts_path.exists():
                                        # 尝试一种回退机制：假设 artifacts 在项目根目录的 .refactor_agent_runs 下
                                        # 提取路径中 .refactor_agent_runs 之后的部分
                                        parts = raw_path.split(".refactor_agent_runs")
                                        if len(parts) > 1:
                                            artifacts_path = root_path / ".refactor_agent_runs" / parts[1].strip(os.sep)
                            except Exception as e:
                                print(f"    ⚠️ 解析 artifacts_dir 失败: {e}")

                # 2. 读取 Artifacts 目录下的详细过程文件
                if artifacts_path and artifacts_path.exists():
                    outfile.write(f"--- [Artifacts Dir] {artifacts_path} ---\n\n")
                    
                    # 获取该目录下所有文件并排序
                    # 排序很重要，为了让 step1, step2 按顺序显示
                    artifact_files = sorted([f for f in artifacts_path.iterdir() if f.is_file()])
                    
                    # 定义我们关心的文件优先级，确保重要的先展示
                    # 比如 plan.json 最先，summary.json 最后，中间是步骤
                    def sort_key(f):
                        name = f.name
                        if "plan.json" in name: return 0
                        if "tool_outputs" in name: return 1
                        if "step" in name: return 2
                        if "summary.json" in name: return 99
                        return 10
                    
                    artifact_files.sort(key=sort_key)

                    for art_file in artifact_files:
                        # 跳过一些不需要的二进制文件或过大的文件
                        if art_file.suffix not in ['.json', '.txt', '.diff', '.log', '.md', '.py', '.java']:
                            continue
                        
                        # 读取内容
                        content = read_file_content(art_file)
                        
                        outfile.write(f"📄 FILE: {art_file.name}\n")
                        outfile.write("-" * 20 + "\n")
                        outfile.write(content)
                        outfile.write("\n" + "-" * 20 + "\n\n")
                else:
                    outfile.write(f"⚠️ Warning: Artifacts directory not found or inaccessible: {artifacts_path}\n")

    print(f"\n✅ 合并完成！文件已保存至: {Path(OUTPUT_FILE).absolute()}")

if __name__ == "__main__":
    main()
