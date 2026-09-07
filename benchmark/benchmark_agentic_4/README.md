# benchmark_agentic_4

独立评测包：评估 **agentic**（超图）在配方相关五类问题上的表现，并与纯 LLM 对照。

## 三步工作流

| mode | 作用 |
|------|------|
| `generate` | 从 `main.db` 专利文档抽样，按题型用 LLM 出题 |
| `evaluate` | 同一题跑 agentic + 纯 LLM，成对 judge，记录详尽延迟指标 |
| `report` | 按题型/整体汇总 JSON |
| `excel` | 导出多 sheet Excel |
| `all` | 依次执行以上全部 |

配置：`benchmark/benchmark_agentic_4/config.yaml`（默认 `run.mode: generate`）。

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate test
cd /root/projects/Chemical-Graph-RAG
python -m benchmark.benchmark_agentic_4
```

## 五类问题（question_type）

1. `product_to_formula` — 由产品（品类+功能）找配方  
2. `application_to_formula` — 由应用需求找配方  
3. `formula_optimize` — 配方优化（痛点+方案）  
4. `ingredient_to_formula_and_others` — 原料 → 含其配方 → 其他原料  
5. `ingredient_interchange` — 原料可互换清单  

数量由 `generate.type_counts` 控制；总和 = 抽取专利篇数（尽量不重复，一篇一题）。

## 文档范围（doc_id_range）

`generate.doc_id_range` 限制抽样用的 `doc.id`（切开后的 doc 表 id，不是原始 PDF 数）：

- `null` / 省略：不限制
- 单个数字 `N`（如 `57032`）：闭区间 **0–N**
- 列表 `[a, b]`：闭区间 **a–b**

示例：

```yaml
generate:
  doc_id_range: null
  # 例如只要大致专利段：doc_id_range: [22952, 57032]
```

不再做 content 关键词 / 专利过滤。

## 评测指标（重点）

agentic 路径至少汇总：

- embed / retrieve / rerank / precompute / rewrite / chunk / node / keyword 等 timing  
- `query_latency_s` / `wall_latency_s` / `retrieve_latency_s`  
- `mean_*` 按 search 次数均摊  
- `n_turns` / `n_search` / `mean_search_s` / `mean_turn_s`  
- recall（有 gold source docs 时）  
- judge 对错、成对胜负  
- 若返回 multi-path timing：原样保留并聚合 `multi_path_n/mean/sum`

## 独立性

禁止 import：`benchmark2` / `benchmark.benchmark_agentic_1` / `benchmark.benchmark_agentic_2` / `useless`。  
可调用被测系统 `src.*` / DHMF。

## 输出

默认目录：`benchmark/benchmark_agentic_4/results/`

- `questions.json`  
- `evals.json`  
- `report.json` / `report.xlsx`  

## 试跑建议

```yaml
run:
  mode: generate
generate:
  type_counts:
    product_to_formula: 1
    application_to_formula: 0
    formula_optimize: 0
    ingredient_to_formula_and_others: 0
    ingredient_interchange: 0
  n_limit: 1
```

无 LLM 凭证时：至少保证 import / AST 隔离 / 文档抽样结构正确，不要硬跑崩 evaluate。
