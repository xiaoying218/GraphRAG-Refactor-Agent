import os
import json
from pathlib import Path

# ================= 配置区域 =================
# 你的 bench_out 输出目录名称
BENCH_OUT_DIR = "bench_out"

# 🔴 关键修改：在这里指定你想合并的任务名称
# 如果想合并所有任务，请留空： TARGET_TASK_NAME = "" 或 TARGET_TASK_NAME = None
# 如果只想合并特定任务，填入名称： TARGET_TASK_NAME = "remove_magic_numbers"
TARGET_TASK_NAME = "remove_magic_numbers"

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
    if TARGET_TASK_NAME:
        print(f"🎯 过滤模式开启：只合并包含 '{TARGET_TASK_NAME}' 的任务")
    else:
        print(f"🔄 全量模式开启：合并所有任务")
    
    found_count = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        outfile.write(f"Refactoring Agent Debug Log\nGenerated Time: {os.times()}\n")
        if TARGET_TASK_NAME:
            outfile.write(f"Filter: Only showing tasks matching '{TARGET_TASK_NAME}'\n\n")
        else:
            outfile.write("Filter: All Tasks\n\n")

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
                
                # --- 🔍 过滤逻辑在这里 ---
                if TARGET_TASK_NAME and (TARGET_TASK_NAME not in task_name):
                    continue
                # -----------------------

                found_count += 1
                print(f"  ✅ Found Task: [{mode_name}] {task_name}")
                
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
                                    artifacts_path = Path(raw_path)
                                    # 回退机制：如果绝对路径找不到，尝试在当前项目下找
                                    if not artifacts_path.exists():
                                        parts = raw_path.split(".refactor_agent_runs")
                                        if len(parts) > 1:
                                            artifacts_path = root_path / ".refactor_agent_runs" / parts[1].strip(os.sep)
                            except Exception as e:
                                print(f"    ⚠️ 解析 artifacts_dir 失败: {e}")

                # 2. 读取 Artifacts 目录下的详细过程文件
                if artifacts_path and artifacts_path.exists():
                    outfile.write(f"--- [Artifacts Dir] {artifacts_path} ---\n\n")
                    
                    artifact_files = sorted([f for f in artifacts_path.iterdir() if f.is_file()])
                    
                    # 排序优先级
                    def sort_key(f):
                        name = f.name
                        if "plan.json" in name: return 0
                        if "tool_outputs" in name: return 1
                        if "step" in name: return 2
                        if "summary.json" in name: return 99
                        return 10
                    
                    artifact_files.sort(key=sort_key)

                    for art_file in artifact_files:
                        if art_file.suffix not in ['.json', '.txt', '.diff', '.log', '.md', '.py', '.java']:
                            continue
                        
                        content = read_file_content(art_file)
                        outfile.write(f"📄 FILE: {art_file.name}\n")
                        outfile.write("-" * 20 + "\n")
                        outfile.write(content)
                        outfile.write("\n" + "-" * 20 + "\n\n")
                else:
                    outfile.write(f"⚠️ Warning: Artifacts directory not found or inaccessible: {artifacts_path}\n")

    if found_count == 0:
        print(f"\n⚠️ 未找到任何包含 '{TARGET_TASK_NAME}' 的任务。请检查名称拼写。")
    else:
        print(f"\n✅ 合并完成！共处理 {found_count} 个任务。文件已保存至: {Path(OUTPUT_FILE).absolute()}")

if __name__ == "__main__":
    main()