import pathlib
import os

# 配置需要忽略的文件夹或文件
IGNORE_LIST = {'.git', '.idea', '__pycache__', 'venv', '.venv', '.vscode', 'node_modules', '.DS_Store', '.refactor_agent_runs','.hypothesis','.pytest_cache','.env'}

def generate_tree(dir_path: pathlib.Path, prefix: str = ""):
    """
    递归生成目录树结构
    :param dir_path: 当前目录路径 (Path对象)
    :param prefix: 当前的前缀字符串 (用于缩进)
    """
    try:
        # 获取当前目录下的所有内容，并根据是否是文件夹和名称排序
        # 排序规则：文件夹优先，然后按字母顺序
        contents = list(dir_path.iterdir())
        contents.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        print(f"{prefix}└── [Permission Denied]")
        return

    # 过滤掉忽略列表中的项
    contents = [item for item in contents if item.name not in IGNORE_LIST]

    # 遍历当前目录下的文件/文件夹
    pointers = [ "├── " ] * (len(contents) - 1) + [ "└── " ]
    
    for pointer, path in zip(pointers, contents):
        # 打印当前项
        print(f"{prefix}{pointer}{path.name}{'/' if path.is_dir() else ''}")

        # 如果是目录，则递归调用
        if path.is_dir():
            # 这里的逻辑是：如果当前项是最后一个 (└──)，子项的前缀就是空格
            # 如果当前项不是最后一个 (├──)，子项的前缀就需要竖线 (│)
            extension = "│   " if pointer == "├── " else "    "
            generate_tree(path, prefix + extension)

if __name__ == "__main__":
    # 获取当前脚本所在的目录作为根目录
    root_dir = pathlib.Path.cwd()
    
    print(f"Project Structure for: {root_dir.name}")
    print(f"Path: {root_dir}")
    print("." )
    
    generate_tree(root_dir)