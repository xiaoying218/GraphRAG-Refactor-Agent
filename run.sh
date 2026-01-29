# 查看目录结构
python generate_tree.py 

# 导出代码
python export_code.py 


source .venv/bin/activate

python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

ps aux | egrep "uvicorn|python.*uvicorn" | grep -v grep

kill -9 52237

