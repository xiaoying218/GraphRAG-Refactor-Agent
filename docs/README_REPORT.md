# Benchmark report (auto-generated)

_Generated at 2026-01-05 04:49:25_

This report aggregates multiple benchmark output folders under the provided `--inputs`.

## Overall (across tasks)

| mode | avg context coverage | pass rate | accept rate | runs |
| --- | --- | --- | --- | --- |
| graph_rag | 41.7% | 100.0% | 100.0% | 5 |
| vector_only | 21.7% | 80.0% | 80.0% | 5 |

## Per-task: GraphRAG vs Vector-only

| task | GraphRAG coverage | GraphRAG pass | GraphRAG accept | GraphRAG failure dist | Vector coverage | Vector pass | Vector accept | Vector failure dist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `consolidate_base_score_computation` | 66.7% | 100.0% | 100.0% | success:1 | 0.0% | 100.0% | 100.0% | success:1 |
| `introduce_region_context` | 0.0% | 100.0% | 100.0% | success:1 | 0.0% | 100.0% | 100.0% | success:1 |
| `refactor_legacy_scoring_util` | 50.0% | 100.0% | 100.0% | success:1 | 50.0% | 0.0% | 0.0% | accept_failed:1 |
| `remove_magic_numbers` | 66.7% | 100.0% | 100.0% | success:1 | 33.3% | 100.0% | 100.0% | success:1 |
| `split_campaign_decision_engine_responsibilities` | 25.0% | 100.0% | 100.0% | success:1 | 25.0% | 100.0% | 100.0% | success:1 |

> **Note on `context coverage`:** this is computed from the produced `context_pack` only. An agent can still succeed if it reads files directly during execution, even when coverage is low.

## Failure taxonomy (counts)

| category | graph_rag | vector_only |
| --- | --- | --- |
| success | 5 | 4 |
| accept_failed | 0 | 1 |

## Case study: `split_campaign_decision_engine_responsibilities` — GraphRAG recovers implicit dependencies

- Vector-only context nodes: **6**
- GraphRAG context nodes: **13**
- GraphRAG-only nodes: **7**

GraphRAG-only nodes (selected):

- `LegacyScoringUtil.calculateBaseScore`  
  - file: `src/main/java/com/icbc/marketing/core/LegacyScoringUtil.java`  
  - roles: callee  
  - why: call-graph neighbor (CALLEE, depth=1)
- `LegacyScoringUtil.isBlacklisted`  
  - file: `src/main/java/com/icbc/marketing/core/LegacyScoringUtil.java`  
  - roles: callee  
  - why: call-graph neighbor (CALLEE, depth=1)
- `EXTERNAL::Map.getOrDefault`  
  - file: ``  
  - roles: callee  
  - why: call-graph neighbor (CALLEE, depth=2)
- `EXTERNAL::String.startsWith`  
  - file: ``  
  - roles: callee  
  - why: call-graph neighbor (CALLEE, depth=2)
- `EXTERNAL::System.out.println`  
  - file: ``  
  - roles: callee  
  - why: call-graph neighbor (CALLEE, depth=1)
- `UNRESOLVED::execute`  
  - file: ``  
  - roles: callee  
  - why: call-graph neighbor (CALLEE, depth=1)
- `UNRESOLVED::isApplicable`  
  - file: ``  
  - roles: callee  
  - why: call-graph neighbor (CALLEE, depth=1)

**Takeaway:** GraphRAG expands along repo-level relations (e.g., CALLER/CALLEE). This tends to surface *implicit* cross-file dependencies (call sites, strategy implementations, engine entry points) that a vector-only “same-file / semantically similar” retrieval can miss.

## Future work

- **Replace heuristic acceptance with tests:** define characterization/unit tests as the success criterion (compile + test suite pass), instead of relying on missing-expected-files / focus-method checks.
- **Test-guided repair loops:** during agent iterations (`--max-iters`), run tests after each edit and use the failing test outputs/stack traces as feedback to guide repair.
- **Stronger evaluation:** add refactoring-specific test cases for each task (including negative/edge cases) and report success rates across multiple seeds.
