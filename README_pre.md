# Graph-RAG Context Engine (Step A)

This folder upgrades your current code-graph demo into a **traceable Graph-RAG Context Engine**:

✅ Each Method/Field/Class node stores:
- `file_path`
- `start_line`, `end_line`
- `signature`
- `docstring` (Javadoc / nearest preceding comment)

✅ Dual index:
- Graph (`networkx`) for structure-aware neighborhood expansion
- Vector index (`TF-IDF`) for semantic seed retrieval

✅ Retrieval strategy:
1) vector search -> seed nodes
2) graph expansion (k-hop CALLS + rule-based expansion)
3) output a structured `context_pack.json` (IDE-like preview)

---

## Run demo

```bash
pip install -r requirements.txt
python demo_context_pack.py --project /path/to/java/repo --query "refactor duplicated offer decision logic"
```

You will get:
- terminal summary
- `context_pack.json` in current directory

---

## How to integrate into your existing repo

Replace or add these files under your `src/`:

- `src/parser.py`
- `src/graph_builder.py`
- `src/vector_index.py`
- `src/context_engine.py`
- `src/utils.py`

Then in your main flow you can do:

```python
from src.parser import JavaCodeParser
from src.graph_builder import CodeGraphBuilder
from src.vector_index import NodeVectorIndex
from src.context_engine import GraphRAGContextEngine

parser = JavaCodeParser()
data = parser.parse_project(PROJECT_PATH)

builder = CodeGraphBuilder()
builder.build_from_parsed_data(data)
graph = builder.get_graph()

vindex = NodeVectorIndex()
vindex.build_from_graph(graph)

engine = GraphRAGContextEngine(graph, vindex)
context_pack = engine.query("reduce complexity in decideOffer", seed_top_k=5, hops=2)
```

---

## Next step (B)

Once you have `context_pack`, the next step is to implement:
- patch generation (diff)
- compile/tests/lint verification loop
- repair iterations

This turns your context engine into a real refactoring agent.
