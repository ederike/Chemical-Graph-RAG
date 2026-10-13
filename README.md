# extra 分支说明

这份说明只存在于 `extra`。`main` 上没有仓库根目录的 `README.md`。打开 GitHub 仓库时如果停在默认的 `main`，看不到本文；把分支切到 `extra` 才能看到。

本文说明三件事：这个分支和代码主线是什么关系，分支里多出来的每一类文件是什么、数字从哪份报告来，以及以后怎样只更新 `extra`、不把这些文件送进阿里云 Codeup 的 `develop`。

材料提交是 `a474bfa`（说明：补充公开仓库用的文档、评测结果和脱敏示例配置）。它的父提交是 `a7051c5`（说明：重构了 oss 下载和 insert 的 md5 正文去重管线）。本文是叠在 `a474bfa` 之上的下一次提交。`a474bfa` 相对 `a7051c5` 新增 81 个文件。代码树和 `main` 的 `a7051c5` 相同。

## 1. 为什么要有 extra

同一份 Git 提交在所有远端上的文件树是一样的。文档、评测结果和示例配置如果写进 `main`，下一次同步就会进 `origin/develop`。这些目录体积大，配置里还有内网地址，开发主线不收它们。

因此把它们放在单独的 `extra` 上：

- 本地 `main`、阿里云 `origin/develop`、GitHub `main` 保持同一个代码提交，之后可以各自快进。
- `docs/`、`results/`、`example/demo/config_open.yaml` 只出现在 `extra` 的历史里。
- 在 GitHub 上浏览这些材料时，打开 `extra`，不要在 `main` 上找。

`extra` 的上游是 `github/extra`。本机这条分支跟踪 GitHub，不跟踪 `origin`。

## 2. 四条引用分别指向哪里

| 位置 | 在本文写入时的提交 | 里面有没有 docs / results / demo 配置 |
| --- | --- | --- |
| 本地 `main` | `a7051c5`，上游 `origin/develop` | 没有。这三个路径写在 `.gitignore` 里 |
| `origin/develop`（阿里云 Codeup） | `a7051c5` | 没有 |
| GitHub `main` | `a7051c5` | 没有 |
| 本地 `extra` 与 GitHub `extra` | `a474bfa`，再加上本文 | 有 |

本地 `main` 的推送配置是 `refs/heads/main:refs/heads/develop`。在 `main` 上点源代码管理的同步，送去的是阿里云的 `develop`，不是 GitHub。GitHub 的 `main` 要另外执行 `git push github main`，而且只有在它和本地 `main` 还是同一条历史时，这次推送才是快进。

`origin` 上没有名为 `extra` 的分支。不要把 `extra` 推到 `origin`。那会在 Codeup 上新建一条带着这些文件的分支。

## 3. 可以做和不要做的事

可以：

- 在 GitHub 上切换到 `extra` 阅读 `docs/`、`results/` 和 `example/demo/config_open.yaml`。
- 在单独的工作区检出 `extra`，只读这些材料。
- 代码仍在 `main` 上提交。提交后的同步更新 `origin/develop`。需要 GitHub 的 `main` 跟着前进时，再 `git push github main`。
- 若要让 `extra` 也带上更新后的代码，在 `extra` 上把 `main` 合并进来，再只推 `github extra`。方向是 `main` 进入 `extra`。

不要做：

- 不要把 `extra` 合并进 `main`。合并结果的文件树里会带上 `docs/`、`results/` 和 demo 配置，下一次同步会把它们送进 `origin/develop`。
- 不要在已经放着本机知识库的开发目录里直接 `git checkout extra`。说明见第 10 节。
- 不要执行 `git push origin extra`，也不要在源代码管理里对 `extra` 选择发布到 `origin`。
- 不要把 `example/a`、`example/b`、`example/c` 或 `docs/jump` 补进这个分支。前三个是本机库和原文，体积约数十 GB；`docs/jump` 含有云存储密钥和登录口令，故意没有收录。

`.gitignore` 在 `extra` 和 `main` 上是同一份。它仍然忽略 `docs/`、`results/` 和整个 `example/`。这些路径能出现在 `extra` 里，是因为当初用 `git add -f` 按路径强制加入。忽略规则只影响尚未跟踪的文件。已经跟踪的文件会继续留在 `extra` 的提交里。在 `extra` 上普通的 `git add .` 不会把 `example/a` 或 `docs/jump` 收进去；若有人使用 `git add -f` 并写上这些路径，就会收进去。更新 `extra` 时只添加打算公开的路径。

## 4. 本分支多出来的内容

相对 `a7051c5`，`a474bfa` 增加三类路径。

### 4.1 `docs/`：15 个已收录文件

收录的是说明、汇报和一张消融表。未收录 `docs/jump`。

| 文件 | 它是什么 |
| --- | --- |
| `docs/API入门.md` | 面向当时那个只有 HTTP 窗口、还没有网站的服务，讲解 URL、JSON、GET/POST，以及 `/health`、`POST /query`、`POST /multihop-query`、`POST /retrieve` |
| `docs/Docker部署.md` | 镜像只装代码和 Python 环境，知识库另拷到服务器再挂载。硬件要求指向部署需求：内存至少 32 GB |
| `docs/部署文档.md` | 不用 Docker 的装法：conda、systemd、nginx。内存不足 32 GB 时文档要求停止 |
| `docs/部署需求.md` | 从「开发机跑 API + 远端反代」改成「一台服务器独立运行」时的机器、软件、网络、数据和验收条件 |
| `docs/部署命令.md` | 几乎是空文件，只有一个字节，没有可执行的命令清单 |
| `docs/需求.md` | 起草 9 月 11 日之后那份工作汇报时留下的写作提示，不是产品需求规格 |
| `docs/agentic原理与实现详解.md` | Agentic 循环、三种工具、检索八步、以及和多跳 Agent 的对照。文内有代码路径索引 |
| `docs/WeKnora与本项目知识库构建与检索对比.md` | 2026-09-22 对腾讯 WeKnora `0.8.0` 与本仓库在解析、分块、图谱、召回、重排上的结构对照 |
| `docs/主流大模型 vs 超图知识图谱系统：图谱必要性验证.md` | 用 200 题闭卷对照，看大模型不检索时能否代替本地库。分析对象写的是 `report_mainstream_20260901_1942.xlsx`，该 xlsx 不在本分支 |
| `docs/精度消融.xlsx` | 向量索引消融。三个工作表。维度组合是 `l2` / `hnsw`，4096、1024、768 维，`fp32` / `fp16`。记录项是准确率、召回率、意外失败数 |
| `docs/两周工作汇报.md` | 2026-08-12 至 08-26。库为 `example/a/DB`，回答模式 Agent，模型 `qwen3.6-27b` |
| `docs/两周工作汇报_20260826-0908.md` | 2026-08-26 至 09-08。对照 Agentic、Agent 规划图和无检索模型 |
| `docs/两周工作汇报_20260911-0922.md` | 2026-09-11 至 09-22。网站、容器、账号会话、MSDS 进度、生产延迟 |
| `docs/工作汇报_20260911-1012.md` | 2026-09-11 至 10-12。库规模、生产延迟、MSDS 自生成题、与 WeKnora 的同题对照。回答和裁判模型写的是 `llm-medium` |
| `docs/化工超图知识图谱系统总结报告.md` | 把库规模、在线站点、延迟、MSDS 正确率、WeKnora 对照收成一篇总报告 |

汇报之间的数字口径不完全相同：模型名、题集范围和日期会变。引用某一次结果时，以对应目录里的 `report.json` 的 `meta.created_at` 为准，不要把两篇汇报里的百分数当成同一次实验。

`docs/API入门.md` 描述的是早期根路径接口。后来的网站和账号系统在 `chemical-rag-web/` 与 `api/`，以代码和 `docs/部署文档.md` 为准。

### 4.2 `results/`：65 个已收录文件

Excel 打开时产生的锁文件 `results/benchmark_agentic_5_test/~$report.xlsx` 没有收录。本机如果还看得到它，那是编辑器残留，不在 Git 里。

各目录的题量、正确率和时延写在第 7 节。读文件时先看第 6 节的字段约定。

### 4.3 `example/`：只有一个文件

`example/demo/config_open.yaml`。

没有 `example/a`、`example/b`、`example/c`。因此本分支没有 SQLite、FAISS、PDF 原文，也不能在只有这份检出的机器上跑通问答。配置里的 `settings.working_path` 仍是 `example/a`，那个目录要在本机自备。

## 5. 和这些材料配套的代码在哪里

下面这些路径 `main` 和 `extra` 都有，因为它们属于 `a7051c5` 的代码树。

对外构建和问答的入口是 `src`。调用方构造 `DHMF`。构建顺序是下载（可选）、识别、总结、分块、抽取、建节点、向量化。三条问答路径各管一种问法：

- `query`：一次检索后生成。问句已经指向某份资料中的一段事实时用。
- `agent_query`：模型先拆子问题，再逐个检索，最后汇总。一句话里有多个必须分别查证的条件时用。
- `agentic_query`：模型自己决定下一轮是再搜、读原文、看同文档实体，还是作答。第一次检索不够时用。

`agent` 和 `agentic` 各读配置里自己的段，不回退到 `retrieve` 的模型地址。向量索引、嵌入和重排是三套问答共用的。

和本分支结果对应的评测代码在：

| 结果目录 | 代码位置 |
| --- | --- |
| `results/benchmark_agentic_1_*` | `benchmark/benchmark_agentic_1/` |
| `results/benchmark_agentic_2_*` | `benchmark/benchmark_agentic_2/` |
| `results/benchmark_agentic_4_专利测试集/` | `benchmark/benchmark_agentic_4/` |
| `results/benchmark_agentic_5_*` | `benchmark/benchmark_agentic_5/` |
| `results/prod_latency_20260922/` | `scripts/prod_latency_bench.py` |
| `results/benchmark_results_sen-analysis-topk/` | 重排 top-k 扫描的产物，汇总表在该目录根上 |

`src/README.md`、`api/README.md`、`chemical-rag-web/README.md` 以及各 benchmark 包里的 README 仍是代码说明。本文不代替它们。

## 6. 评测文件怎么读

一次评测通常拆成四类文件。某个目录缺哪一类，就表示那一类没有做完或没有放进本分支。

| 文件 | 作用 |
| --- | --- |
| `questions.json` 或 `set*_dataset.json` | 题目。外层有 `meta` 和 `questions`。`meta` 记录生成时间、种子、来源库、题量和模型 |
| `evals.json` | 逐题结果。外层是 `meta`、`results`、`summary`。每题含问题、标准答案、来源、召回，以及各系统的回答和裁判比较 |
| `report.json` | 汇总。常见外层是 `meta` 与 `summary`。正确率在对应系统的 `llm_acc`，时延在 `latency`，文档召回在 `doc_recall`，胜负在 `comparison` |
| `report.xlsx` | 同一份汇总的表格，方便直接打开。和 `report.json` 冲突时，以 `report.json` 为准 |
| `set*_stats.json` | 题集设计说明和题目分布，不是回答正确率 |

`llm_acc.accuracy` 是「评判为正确的题数 / 进入裁判的题数」。有的运行会比原题少 1 题，因为有一题没有完成评判。引用时同时写 `n_correct` 和 `n_judged`。

`comparison.win` 是成对胜负：检索系统胜、对照系统胜、平局。`both_correct`、`agentic_only`、`llm_only_only` 或 `weknora_only`、`both_wrong` 是对错的交叉，和胜负不是同一个数。胜负优先采用裁判给出的 `comparison.winner`；裁判没有给时，按「正确优于错误」补。

`doc_recall.mean_recall` 是文档命中的平均召回，不是正确率。只做 MSDS 的那次运行里，正确率高而平均召回低，两套数字都按报告原样保留。

`latency.mean_query_s` 是端到端平均秒数，`mean_retrieve_s` 是检索平均秒数。更细的预计算、改写、嵌入、chunk、node、keyword、扩展、重排都在 `latency` 的分项里。下文只摘平均端到端和平均检索；分项去对应的 `report.json`。

JSON 和 CSV 的 `meta` 里常有开发机绝对路径，例如 `/root/projects/Chemical-Graph-RAG/...`。那是生成报告时写下的路径，不是要求读者在自己的机器上建同样的目录。

## 7. 各份结果

百分数由报告里的 `n_correct / n_judged` 四舍五入到一位小数。原始小数仍在 JSON 里。

### 7.1 `results/benchmark_agentic_1_含专利/`

库内自生成题，范围含专利语料。题目文件生成于 2026-08-22，模型 `qwen3.6-27b`，种子 44，成功 200 题。跳数计划是 1 跳 100、2 跳 50、3 跳 50。来源库路径记的是 `example/a/DB/main.db`。

报告生成于 2026-09-03。模式 `agentic`，裁判 `qwen3.6-27b`，pairwise，200 题全部进入评判，耗时约 5888 秒。

| 系统 | 正确 / 评判 | 准确率 | 平均端到端 | 平均检索 | 平均文档召回 |
| --- | ---: | ---: | ---: | ---: | ---: |
| agentic | 192/200 | 96.0% | 46.3 秒 | 14.1 秒 | 0.847 |
| 纯模型，不检索 | 20/200 | 10.0% | 29.5 秒 | 无 | 无 |

成对 200 题：agentic 胜 192，纯模型胜 6，平局 2。两边都对 19，仅 agentic 对 173，仅纯模型对 1，两边都错 7。

文件：`questions.json`、`evals.json`、`report.json`、`report.xlsx`。

### 7.2 `results/benchmark_agentic_1_仅msds/`

同一套自生成流程，文档范围改成 MSDS。题目生成于 2026-10-01，模型 `llm-medium`，`doc_id` 57033–257613，来源文档数 200577，种子 44，成功 200。跳数计划仍是 100 / 50 / 50。

报告生成于 2026-10-02。裁判 `llm-medium`。200 题里 199 题进入评判，评测耗时约 111206 秒。

| 系统 | 正确 / 评判 | 准确率 | 平均端到端 | 平均检索 | 平均文档召回 |
| --- | ---: | ---: | ---: | ---: | ---: |
| agentic | 188/199 | 94.5% | 512.6 秒 | 480.6 秒 | 0.268 |
| 纯模型，不检索 | 36/199 | 18.1% | 14.0 秒 | 无 | 无 |

成对 199 题：agentic 胜 192，纯模型胜 7，无平局。两边都对 33，仅 agentic 对 155，仅纯模型对 3，两边都错 8。

这次平均端到端远高于 9 月 3 日那次。报告里的 `latency.mean_node_s` 约为 449.8 秒，时间主要记在节点向量检索上。和「含专利」那次比较时，要同时看日期、`doc_id` 范围和这份时延分项，不能只比两个正确率。

文件与上一节相同的四件套。

### 7.3 `results/benchmark_agentic_2_questions1_agentic_new/`

外部题集「建筑涂料测试问题集 1」。`set1_stats.json` 的标题是「建筑涂料Graph-RAG测试问题集 - 统计概览」，来源表名 `建筑涂料测试问题集1.xlsx`，题目 1177，整理时间 2026-09-04。分布字段包括难度等级、查询类型、问题维度、推理跳数、知识层，以及材料类别、产品类别、应用场景。

`set1_report.json` 生成于 2026-09-05，`n_results` 1177，实际成对评判 1176。这一套报告没有填写文档召回。

| 系统 | 正确 / 评判 | 准确率 | 平均端到端 | 平均检索 |
| --- | ---: | ---: | ---: | ---: |
| agentic | 1014/1176 | 86.2% | 165.6 秒 | 35.0 秒 |
| 纯模型，不检索 | 720/1176 | 61.2% | 56.2 秒 | 无 |

成对胜负：agentic 胜 950，纯模型胜 207，平局 19。agentic 胜率约 80.8%。两边都对 590，仅 agentic 对 424，仅纯模型对 130，两边都不对 32。平均分差（agentic 减纯模型）约为 0.881。交叉表在 `summary.comparison.crosstab`，里面把「正确 / 部分正确 / 错误」分开计数，所以和上面的「仅 agentic 对 424」不是同一层汇总。

文件：`set1_dataset.json`（1177 题）、`set1_stats.json`、`set1_evals.json`（约 63 MB，超过 GitHub 建议的 50 MB，低于 100 MB 拒绝线）、`set1_report.json`、`set1_report.xlsx`。

### 7.4 `results/benchmark_agentic_2_questions2_agentic_new/`

外部题集「建筑涂料测试问题集 2」。只有题目和设计统计，没有 `evals`、`report` 或 xlsx。本分支不能从这里读出正确率。

`set2_dataset.json`：1020 题，整理于 2026-09-07，工作表名「测试集(1000+)」。

`set2_stats.json` 的设计说明写明：

- 用途是比较「行业超图 + LLM」和「纯 LLM」在建筑涂料问题上的差别。
- 数据基础写成约 10 万份技术文档（原材料 TDS/MSDS、建筑涂料配方）。
- 提问者身份 5 类：原材料生产商、原材料贸易商、涂料厂技术、涂料厂采购、涂料厂质检与应用。
- 问题层面分原材料和制品。
- 能力维度 11 类：单跳事实检索、属性对比、条件筛选、应用匹配、多跳推理、配方分析、问题诊断、选型推荐、成本计算、法规合规、交叉验证。
- 难度 L1 到 L4，对应从单文档事实到多约束综合。
- 免责声明：题集用于评测对照，不构成配方或采购决定。

### 7.5 `results/benchmark_agentic_4_专利测试集/`

专利配方题。题目生成于 2026-09-04，模型 `qwen3.6-27b`，种子 42，`doc_id` 范围 33883–57032，来源文档 23150。计划 100 题，成功 99，失败 1。五类计划各 20 题：`product_to_formula`、`application_to_formula`、`formula_optimize`、`ingredient_to_formula_and_others`、`ingredient_interchange`。`questions.json` 里保留了 100 条记录，其中含失败项；报告按 99 条评判。

报告生成于 2026-09-04 23:33，裁判 `qwen3.6-27b`，评测耗时约 19445 秒。

| 系统 | 正确 / 评判 | 准确率 | 平均端到端 | 平均检索 | 平均文档召回 |
| --- | ---: | ---: | ---: | ---: | ---: |
| agentic | 91/99 | 91.9% | 285.7 秒 | 178.0 秒 | 0.788 |
| 纯模型，不检索 | 4/99 | 4.0% | 87.6 秒 | 无 | 无 |

成对 99 题：agentic 胜 90，纯模型胜 4，平局 5。两边都对 3，仅 agentic 对 88，仅纯模型对 1，两边都错 7。分类型数字在 `summary.agentic.by_question_type`。

文件是 questions、evals、report.json、report.xlsx 四件套。

### 7.6 `results/benchmark_agentic_5_test/`

小样本试跑，用来接通 WeKnora 对照，不是 200 题正式结果。题目生成于 2026-09-30，库路径记的是 `example/b/DB/main.db`，`original_pdf_only` 为真，来源文档 298，种子 42，成功 12。跳数是 1 跳 10、2 跳 1、3 跳 1。

报告同日 15:27。裁判 `llm-medium`。WeKnora 知识库名记为 `test`，对话模式 `agent`。12 题全部评判，评测耗时约 236 秒。

| 系统 | 正确 / 评判 | 准确率 | 平均端到端 | 平均检索 | 平均文档召回 |
| --- | ---: | ---: | ---: | ---: | ---: |
| agentic | 12/12 | 100% | 69.3 秒 | 54.7 秒 | 0.917 |
| WeKnora | 4/12 | 33.3% | 6.0 秒 | 0.49 秒 | 0.667 |

12 题 agentic 全胜，无平局。两边都对 4，仅 agentic 对 8。样本只有 12，不能拿来代替下一节的 200 题。

### 7.7 `results/benchmark_agentic_5_与weknora对比/`

与上一节相同的出题约束：`example/b` 的库、只使用原始 PDF、来源文档 298、种子 42、模型 `llm-medium`。题目 200，生成于 2026-09-30 17:17。跳数计划在报告 `meta.hop_counts`。

报告评判了 199 题。

| 系统 | 正确 / 评判 | 准确率 | 平均端到端 | 平均检索 | 平均文档召回 |
| --- | ---: | ---: | ---: | ---: | ---: |
| agentic | 195/199 | 98.0% | 26.4 秒 | 9.4 秒 | 0.859 |
| WeKnora | 63/199 | 31.7% | 6.7 秒 | 0.33 秒 | 0.597 |

成对 199 题：agentic 胜 196，WeKnora 胜 1，平局 2。两边都对 63，仅 agentic 对 132，仅 WeKnora 对 0，两边都错 4。

WeKnora 更快，正确率和文档召回都低。这是这份报告里的对照，不是把两个系统的时延换成同一硬件后再测的结论。知识库名、对话模式和地址在 `report.json` 的 `meta`。

文件是 questions、evals、report.json、report.xlsx。

### 7.8 `results/benchmark_results_sen-analysis-topk/`

重排保留条数 `rerank_top_k` 从 5 扫到 30，共 26 档，没有缺档。题目是 `questions_complex.json`：200 题，生成于 2026-08-28，模型 `qwen3.6-27b`，种子 44，范围记为 TDS，`max_doc_id` 25350，`chunk_max_vectors` 126911，来源库 `example/a/DB/main.db`。跳数计划在该文件的 `meta.hop_counts`。

每一档一个子目录 `benchmark_result_topk=<k>/evals_complex.json`，没有单独的 `report.json`。汇总在目录根：

- `rerank_topk_sweep_summary.csv`
- `rerank_topk_sweep_summary.json`
- `rerank_topk_sweep_summary.xlsx`
- `rerank_topk_acc_recall.png`
- `rerank_topk_latency.png`

CSV 里 `k=22` 的评判数是 193，其余档是 200。准确率用该档自己的 `n_correct / n_judged`。平均召回随 k 从 66.2% 升到约 88%，平均端到端从 90.9 秒升到 168.8 秒。平均检索大多在 11–13 秒，平均重排在 2.1–2.7 秒。正确率不是随 k 单调上升：k=20 为 194/200（97.0%），k=30 为 187/200（93.5%）。

| k | 评判数 | 正确 | 准确率 | 平均召回 | 平均端到端秒 | 平均检索秒 | 平均重排秒 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 200 | 165 | 82.5% | 66.2% | 90.9 | 11.6 | 2.20 |
| 6 | 200 | 165 | 82.5% | 69.2% | 106.5 | 11.9 | 2.12 |
| 7 | 200 | 171 | 85.5% | 71.2% | 108.4 | 11.9 | 2.19 |
| 8 | 200 | 173 | 86.5% | 74.2% | 103.8 | 11.1 | 2.16 |
| 9 | 200 | 180 | 90.0% | 76.3% | 110.9 | 11.8 | 2.48 |
| 10 | 200 | 176 | 88.0% | 76.6% | 113.0 | 11.0 | 2.15 |
| 11 | 200 | 175 | 87.5% | 78.3% | 116.7 | 11.1 | 2.15 |
| 12 | 200 | 176 | 88.0% | 80.1% | 115.6 | 11.3 | 2.22 |
| 13 | 200 | 184 | 92.0% | 79.6% | 121.6 | 11.6 | 2.23 |
| 14 | 200 | 182 | 91.0% | 81.2% | 124.3 | 11.4 | 2.45 |
| 15 | 200 | 184 | 92.0% | 82.8% | 129.2 | 12.1 | 2.50 |
| 16 | 200 | 189 | 94.5% | 81.8% | 129.7 | 11.5 | 2.17 |
| 17 | 200 | 188 | 94.0% | 83.5% | 131.4 | 11.4 | 2.49 |
| 18 | 200 | 191 | 95.5% | 84.8% | 135.6 | 11.6 | 2.23 |
| 19 | 200 | 183 | 91.5% | 84.1% | 138.1 | 11.5 | 2.26 |
| 20 | 200 | 194 | 97.0% | 85.7% | 136.7 | 11.5 | 2.20 |
| 21 | 200 | 189 | 94.5% | 88.6% | 147.0 | 13.2 | 2.23 |
| 22 | 193 | 183 | 94.8% | 87.2% | 146.4 | 13.0 | 2.21 |
| 23 | 200 | 184 | 92.0% | 85.3% | 143.3 | 13.4 | 2.69 |
| 24 | 200 | 186 | 93.0% | 85.8% | 150.0 | 11.9 | 2.59 |
| 25 | 200 | 187 | 93.5% | 86.1% | 152.5 | 12.7 | 2.66 |
| 26 | 200 | 185 | 92.5% | 85.8% | 159.0 | 11.6 | 2.68 |
| 27 | 200 | 188 | 94.0% | 88.1% | 152.8 | 10.5 | 2.49 |
| 28 | 200 | 191 | 95.5% | 87.1% | 160.1 | 11.4 | 2.49 |
| 29 | 200 | 189 | 94.5% | 86.9% | 164.1 | 12.1 | 2.55 |
| 30 | 200 | 187 | 93.5% | 88.1% | 168.8 | 12.1 | 2.70 |

### 7.9 `results/prod_latency_20260922/`

2026-09-22 对当时在线站点做的延迟抽样，不是正确率评测。`summary.json` 记录入口 `http://182.92.85.174:8060`，种子 20260922，每个题集 30 题，共 90 题，并发 3，单题超时 900 秒。开始 09:40:34，结束 10:17:33，墙钟 2218.64 秒。成功 89，失败 1，失败落在建筑涂料题集那 30 题里。

抽样来源写在 `sample.json`：

| set_id | 名称 | 题目来源 |
| --- | --- | --- |
| `agentic_1` | 库内自生成试题 | `results/benchmark_agentic_1_含专利/questions.json` |
| `coatings_set1` | 建筑涂料测试问题集 1 | `results/benchmark_agentic_2_questions1_agentic_new/set1_dataset.json` |
| `patent` | 专利配方试题 | `results/benchmark_agentic_4_专利测试集/questions.json` |

`results.jsonl` 有 90 行，一行一题。字段包括题集、问题、跳数或题型、客户端总时间、首事件时间、服务端时延、检索时延、轮数、来源数、回答长度、token，以及 precompute、rewrite、embed、chunk、node、keyword、expand、rerank。`sample.json` 只是抽中的题，没有时延。

`生产端速度延迟测试.xlsx` 有两个工作表：「汇总」和「逐题结果」。

下表单位是秒，来自 `summary.json` 的 `groups`。`n` 是进入统计的请求数。

| 分组 | n | 成功 | 客户端总时间中位 / P90 | 首事件中位 / P90 | 服务端中位 / P90 | 检索中位 / P90 |
| --- | ---: | ---: | --- | --- | --- | --- |
| 全部 | 90 | 89 | 61.66 / 125.59 | 17.28 / 29.70 | 61.42 / 125.23 | 6.72 / 17.73 |
| 库内自生成试题 | 30 | 30 | 44.02 / 62.72 | 15.45 / 25.69 | 43.76 / 62.51 | 7.31 / 17.13 |
| 建筑涂料测试问题集 1 | 30 | 29 | 89.30 / 175.85 | 17.32 / 32.63 | 89.08 / 175.65 | 7.02 / 21.63 |
| 专利配方试题 | 30 | 30 | 73.13 / 109.70 | 18.10 / 29.85 | 68.62 / 109.36 | 6.15 / 12.66 |

全部 90 题的客户端总时间平均 73.35 秒，最大 278.02 秒。平均轮数 4.28，平均来源数 18.31，平均回答长度 1603 字。分组的平均轮数、来源数和回答长度在 `groups.<名称>` 的标量字段里。

这次抽样测的是当时线上范围。它和 10 月仅 MSDS 那次 512 秒的正确率评测不是同一套运行，不要把两个平均时间写成一次实验的两个指标。

### 7.10 `results/temp_专利/`

21 道专利相关问题的小清单，不是第 7.5 节那 99 题的报告。

- `patent_issues_20260901.json`：每项有 `ID` 和 `问题`。`ID` 形如 `complex_q_1hop_0058`。
- `patent_issues_20260901_ans.json`：同样的 21 项，多了 `answer`。

这里没有裁判分、没有召回、没有时延汇总。

## 8. 脱敏配置 `example/demo/config_open.yaml`

文件从本机的 `example/a/config_open.yaml` 复制后删掉了密钥，再放进本分支。本分支这一份以提交 `a474bfa` 里的内容为准。开发机工作区里的同名文件若之后又改过，那份改动被 `example/` 的忽略规则挡住，不会自动出现在 `extra` 上。

发布进 `a474bfa` 时清掉的是三处凭证：

- `ali_oss.*.ACCESS_KEY` 为空字符串
- `ali_oss.*.SECRET_KEY` 为空字符串
- `dm_data_mysql.password` 为空字符串

各段的 `api_key` 在来源文件里已经是 `EMPTY`，发布文件保持 `EMPTY`。没有把可用的模型密钥写进本分支。

仍然保留、需要当内部信息看的字段：

- 各段 `base_url`、`embedding_base_url`、`rerank_base_url`
- `doc.recognition` 里的 `paddleocr_url` 和 `paddleocr_layout_url`
- `dm_data_mysql` 的 `db`、`host`、`port`、`user`（密码为空）
- `ali_oss` 的 `END_POINT` 和 `BUCKET_NAME`（两个密钥为空）
- `settings.working_path: example/a`
- `search_range` 里的 `node_max_vectors` 与 `chunk_max_vectors`，以及注释中的 TDS、专利、MSDS 文档编号范围

因此这份 YAML 能让人看清配置有哪些段，不能拿来直接连上生产库或对象存储。要在本机跑，仍使用本机那份完整配置，并通过环境变量 `DHMF_CONFIG` 指到它。不要把本机完整配置提交到任何分支。

顶层段和它们控制的事：

| 段 | 作用 |
| --- | --- |
| `settings` | 工作目录、调试开关、默认接口地址。`working_path` 决定 SQLite、FAISS、原文和日志放在哪 |
| `app_config` | 进程内的线程数、检索正文截断、CORS、监听地址和端口 |
| `oss_download` | 从对象存储拉原文时的表、文件类型、目录和线程数。密钥在 `ali_oss`，这里不重复放钥匙 |
| `doc` | PDF 识别。包含是否数 token、缓存、DPI、页数上限，以及 VLM 与 PaddleOCR 两路 |
| `summary` | 每份文档一条总结。有自己的模型地址，不借用别的段 |
| `chunk` | 切成头块和正文块 |
| `extract` | 从块里抽实体 JSON |
| `build` | 实体写成图节点 |
| `vectorization` | 把块和节点嵌进向量索引 |
| `retrieve` | 单次检索问答用的模型、嵌入和重排 |
| `agent` | 先拆子问题再检索的那条路径，读本段，不回退到 `retrieve` |
| `agentic` | 工具循环。搜索、读原文、图邻居、证据槽、旧工具结果压缩、重排条数都在这里 |
| `search_range` | 限定节点向量和块向量的下标范围，从而限定 TDS、专利、MSDS 中实际加载的部分 |
| `dm_data_mysql` | 业务库连接。本分支密码为空 |
| `ali_oss` | 对象存储连接。本分支 AccessKey 与 Secret 为空 |

`agentic` 段里和实现说明对应的几个开关：最近若干轮工具结果保留全文，更早的长结果只留头尾，再早的只留占位；证据槽有上限，满了丢最旧。具体数字以 YAML 为准，`docs/agentic原理与实现详解.md` 讲的是这些开关在循环里的位置。

## 9. 本分支解决不了的事

检出 `extra` 得不到：

- `example/a`、`example/b`、`example/c` 下的 PDF、SQLite 和 FAISS
- 能登录对象存储或 MySQL 的密钥
- `docs/jump` 里的传输命令
- 建筑涂料题集 2 的模型回答和分数
- `docs/主流大模型 vs 超图知识图谱系统：图谱必要性验证.md` 所分析的那份 `report_mainstream_20260901_1942.xlsx`

构建或问答仍要在有库的机器上，用本机完整配置运行。网站入口、账号和部署步骤以 `docs/部署文档.md`、`docs/Docker部署.md` 和 `chemical-rag-web/README.md` 为准。

## 10. 在开发目录里怎么看 extra

开发目录里通常已经有被忽略的 `docs/`、`results/` 和 `example/`。`example/a` 是知识库，不能为了看这个分支被换掉或删掉。

在这个开发目录里直接 `git checkout extra` 会碰到两件事：

- `extra` 要写入的 `docs/`、`results/` 和 `example/demo/config_open.yaml` 在当前目录里是忽略文件。Git 可能拒绝覆盖，也可能把已跟踪版本写到这些路径上。
- 再切回 `main` 时，Git 会删掉「只在 `extra` 上跟踪、`main` 上不存在」的文件。本机若只靠工作区留着 `docs/` 和 `results/`，这一步会把已跟踪的那些副本从磁盘上拿走。`example/a` 没有被 `extra` 跟踪，切换分支本身不会按「取消跟踪」去删它；不要为了消冲突而手工删 `example/`。

要阅读或更新 `extra`，另开工作区：

```bash
git fetch github
git worktree add /tmp/cgr-extra extra
```

看完后：

```bash
git worktree remove /tmp/cgr-extra
```

开发目录保持停在 `main`。

只克隆 GitHub、本机没有知识库时，可以直接：

```bash
git clone --branch extra https://github.com/ederike/Chemical-Graph-RAG.git
```

这个克隆没有 `origin` 那种指向 Codeup 的上游，适合阅读。不要在这个克隆里把 `extra` 合并进 `main` 再推回 Codeup。

## 11. 以后只更新 extra

代码提交仍在 `main` 上完成，并按原来的同步进入 `origin/develop`。更新 GitHub 代码主线用 `git push github main`。这两步都不经过 `extra`。

只改材料、不改代码时，在 `extra` 的单独工作区里修改，强制加入本来被忽略的路径，提交后只推 GitHub：

```bash
git add -f docs/某新文件.md results/某新目录 example/demo/config_open.yaml
git commit
git push github HEAD:extra
```

`git add -f` 的路径要逐个写清。不要 `git add -f example` 或 `git add -f docs`，否则会把 `example/a` 或 `docs/jump` 卷进去。提交前用 `git diff --cached --name-only` 看清单。清单里出现 `example/a/`、`example/b/`、`example/c/` 或 `docs/jump` 时，先从暂存区拿掉再提交。

要让材料分支跟上新的代码提交：在 `extra` 工作区执行 `git merge main`，解决冲突后 `git push github HEAD:extra`。合并方向保持为 `main` 进入 `extra`。不要到 `main` 上执行 `git merge extra`。

推送前可以核对远端，确认动的是 GitHub 的 `extra`，`origin/develop` 和 GitHub `main` 的提交号没有被这次推送改掉：

```bash
git ls-remote github refs/heads/extra refs/heads/main
git ls-remote origin refs/heads/develop
```

`git push` 不带参数时，`extra` 的上游是 `github/extra`，会推到 GitHub。这个默认值只在当前分支真的是 `extra` 时成立。在 `main` 上不带参数的推送会按上游和 `remote.origin.push` 进入 `origin/develop`。

## 12. 体积

`results/` 里最大的单文件是 `benchmark_agentic_2_questions1_agentic_new/set1_evals.json`，约 63 MB。GitHub 在推送时提示过它超过建议的 50 MB，并接受了这次推送。单个文件达到 100 MB 时 GitHub 会拒绝。以后往 `extra` 加大文件前先看体积。

本分支不使用 Git LFS。大的 JSON 就是普通 Git 对象。克隆 `extra` 会把这些对象一起拉下来。
