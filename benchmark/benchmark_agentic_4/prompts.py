"""benchmark_agentic_4 自有提示词：配方相关五类问题生成 + 评测裁判。"""

Benchmark_PROMPT = {}

Benchmark_PROMPT["QUESTION_GEN_SYSTEM"] = (
    "你是化工配方/专利资料领域的测试集命题专家。"
    "根据给定的单篇专利文档，按指定题型生成一道可核对的客户场景问题与标准答案。"
    "只输出合法 JSON，不要 Markdown 代码块。"
    "答题模型看不到本次出题用的原始文档；问题里禁止用「文档1」「资料」等排版编号指代资料。"
)

_COMMON_RULES = """
通用约束：
1. 扮演终端客户/配方工程师，问题要具体、有业务场景。
2. 问题与标准答案必须能从给定专利文档中找到依据；不要编造文档没有的组分、配比或结论。
3. 文档未给出的写「文档未明确」。
4. 禁止出现「文档1」「Document 1」「上述专利全文」等出题排版指代；用产品名、品类、功能、原料名、专利主题等客户可独立理解的对象。
5. gold_answer 用中文，简明完整；可含关键配比/条件。
6. explanation 说明依据，点名专利中的产品/体系/原料，不要用序号指代资料。
7. 只输出一个 JSON 对象。
"""

Benchmark_PROMPT["QUESTION_GEN_product_to_formula"] = """
题型：product_to_formula —— 由产品找配方（通过品类 + 功能找）。

任务：根据以下 1 篇专利文档，生成一道「已知产品品类与期望功能，询问可用配方」的问题。

出题要求：
- 问题应给出品类（如水性木器漆、环氧底漆、UV 油墨等）与 1-2 个功能诉求（附着力、耐候、消泡、固化速度等），要求给出配方或起始配方思路。
- 不要在问题中直接给出完整配方表；应让答题方从知识库检索得到。
- gold_answer 应概括该专利中的配方组成（关键原料 + 大致作用/配比若有）。
- entities 至少包含：product_category、functions（列表）、formula_name_or_theme（若有）。

""" + _COMMON_RULES + """
JSON schema：
{{
  "question": "客户问题（中文）",
  "gold_answer": "标准答案（中文）",
  "explanation": "依据说明（中文）",
  "question_type": "product_to_formula",
  "entities": {{
    "product_category": "品类",
    "functions": ["功能1", "功能2"],
    "formula_name_or_theme": "配方/体系主题"
  }}
}}

----- 专利文档 -----
{docs_block}

Output:
"""

Benchmark_PROMPT["QUESTION_GEN_application_to_formula"] = """
题型：application_to_formula —— 由应用需求找配方。

任务：根据以下 1 篇专利文档，生成一道「描述应用场景/基材/工况需求，询问适用配方」的问题。

出题要求：
- 问题侧重应用需求（基材、施工方式、环境、性能指标），而非直接点名某个商品牌号。
- gold_answer 给出专利中对应的配方或关键组成与适用说明。
- entities 至少包含：application、substrate（若有）、key_requirements（列表）。

""" + _COMMON_RULES + """
JSON schema：
{{
  "question": "客户问题（中文）",
  "gold_answer": "标准答案（中文）",
  "explanation": "依据说明（中文）",
  "question_type": "application_to_formula",
  "entities": {{
    "application": "应用场景",
    "substrate": "基材或对象",
    "key_requirements": ["需求1", "需求2"],
    "formula_name_or_theme": "配方/体系主题"
  }}
}}

----- 专利文档 -----
{docs_block}

Output:
"""

Benchmark_PROMPT["QUESTION_GEN_formula_optimize"] = """
题型：formula_optimize —— 配方优化（存在痛点 + 解决方案）。

任务：根据以下 1 篇专利文档，生成一道「当前配方/体系存在痛点，询问如何优化」的问题。

出题要求：
- 问题需同时包含：现有体系/痛点现象 + 期望改进方向。
- gold_answer 应体现专利给出的解决方案（改哪些组分、加什么助剂、工艺调整等）。
- entities 至少包含：pain_point、solution_summary、related_ingredients（列表，可空）。

""" + _COMMON_RULES + """
JSON schema：
{{
  "question": "客户问题（中文）",
  "gold_answer": "标准答案（中文）",
  "explanation": "依据说明（中文）",
  "question_type": "formula_optimize",
  "entities": {{
    "pain_point": "痛点",
    "solution_summary": "解决方案摘要",
    "related_ingredients": ["原料1", "原料2"],
    "formula_name_or_theme": "配方/体系主题"
  }}
}}

----- 专利文档 -----
{docs_block}

Output:
"""

Benchmark_PROMPT["QUESTION_GEN_ingredient_to_formula_and_others"] = """
题型：ingredient_to_formula_and_others —— 根据原料找包含其的配方，再找该配方可能需要的其他原料。

任务：根据以下 1 篇专利文档，生成一道「给定一种原料，询问它常出现在哪些配方，以及该配方还需要哪些其他原料」的问题。

出题要求：
- 问题应点名一种专利中的关键原料（化学名/通用名），询问：1) 包含它的配方；2) 同配方中其他重要原料。
- gold_answer 需覆盖「配方定位 + 其他原料清单/作用」。
- entities 至少包含：seed_ingredient、formula_name_or_theme、other_ingredients（列表）。

""" + _COMMON_RULES + """
JSON schema：
{{
  "question": "客户问题（中文）",
  "gold_answer": "标准答案（中文）",
  "explanation": "依据说明（中文）",
  "question_type": "ingredient_to_formula_and_others",
  "entities": {{
    "seed_ingredient": "起始原料",
    "formula_name_or_theme": "配方/体系主题",
    "other_ingredients": ["其他原料1", "其他原料2"]
  }}
}}

----- 专利文档 -----
{docs_block}

Output:
"""

Benchmark_PROMPT["QUESTION_GEN_ingredient_interchange"] = """
题型：ingredient_interchange —— 原料可互换清单（专利自带）。

任务：根据以下 1 篇专利文档，生成一道「询问某类原料可互换/可替代清单」的问题。

出题要求：
- 优先利用专利中明确写出的可替代、同系物、等价原料、优选/可用范围列表。
- 若专利没有显式互换表，则基于「可用原料范围」概括可互换集合，并在 explanation 标明是范围概括。
- entities 至少包含：ingredient_class、interchange_list（列表）、anchor_ingredient（可选）。

""" + _COMMON_RULES + """
JSON schema：
{{
  "question": "客户问题（中文）",
  "gold_answer": "标准答案（中文）",
  "explanation": "依据说明（中文）",
  "question_type": "ingredient_interchange",
  "entities": {{
    "ingredient_class": "原料类别",
    "anchor_ingredient": "锚点原料",
    "interchange_list": ["原料A", "原料B", "原料C"]
  }}
}}

----- 专利文档 -----
{docs_block}

Output:
"""

Benchmark_PROMPT["PURE_LLM_SYSTEM"] = (
    "你是化工/涂料配方领域的通用技术助手，当前没有产品手册或专利原文可查。"
    "拿不准的具体牌号、精确配比不要编造，直接说明视具体体系/专利配方而定。"
    "不要假装引用了某份专利权利要求或内部配方表。"
)

Benchmark_PROMPT["PURE_LLM_USER"] = """
问题：
{question}

请直接根据自身知识作答。先给常见结论和适用边界，再补注意点。
没有资料时不要编造具体商品的出厂指标或精确专利配比。
"""

Benchmark_PROMPT["JUDGE_SYSTEM"] = (
    "你是严谨的配方问答评测裁判。根据问题、标准答案、完整参考文档与系统回答，"
    "判断是否答到了要点。若同时给出两份回答，必须放在同一标准下对比后分别判定。"
    "只输出合法 JSON。"
)

Benchmark_PROMPT["JUDGE_USER"] = """
请同时评判「回答 A」和「回答 B」是否正确回答了「问题」，并比较哪一份更好。

评判规则（每份回答各自二选一，只能用下列中文标签）：
- 正确：抓住了问题要点，关键配方/原料/方案与标准答案及参考文档相似或等价；允许措辞不同、详略不同，也允许推荐不是标准答案的等价配方只要符合诉求即可，答到要点上即对。
- 错误：答非所问或完全未触及要点，只有空泛类型没有具体配方/原料/方案。

注意：
- 不要要求与标准答案一字不差。
- 不要因为漏掉次要细节就判错误，以「是否答到核心点并实际具体」为准。
- 只做回答层面的评判，不对回答的依据来源做评判。
- 两份回答用同一把尺子：先各自判定对错，再比较相对质量。
- 若一份明显更贴要点、更具体，better 必须标出更好的那一份，不能标 tie。
- 质量接近、对错结论相同才允许 tie。
- 不要猜测哪一份来自检索或哪一份来自模型，只根据内容评判。

只输出 JSON：
{{
  "answer_a": {{"judgment": "正确|错误", "reason": "简短理由（中文，1-3句）"}},
  "answer_b": {{"judgment": "正确|错误", "reason": "简短理由（中文，1-3句）"}},
  "better": "A|B|tie",
  "reason": "对比一句：为什么这一份更好或为何打平"
}}

----- 问题 -----
{question}

----- 标准答案 -----
{ground_truth_answer}

----- 全部相关参考文档（完整原文） -----
{source_block}

----- 回答 A -----
{answer_a}

----- 回答 B -----
{answer_b}

Output：
"""

Benchmark_PROMPT["JUDGE_USER_SINGLE"] = """
请评判「系统回答」是否正确回答了「问题」。

评判规则（二选一，只能用下列中文标签）：
- 正确：抓住了问题要点，关键配方/原料/方案与标准答案及参考文档相似或等价。
- 错误：答非所问或完全未触及要点。

只输出 JSON：
{{
  "judgment": "正确|错误",
  "reason": "简短理由（中文，1-3句）"
}}

----- 问题 -----
{question}

----- 标准答案 -----
{ground_truth_answer}

----- 全部相关参考文档（完整原文） -----
{source_block}

----- 系统回答 -----
{rag_answer}

Output：
"""

QUESTION_GEN_PROMPT_KEYS = {
    "product_to_formula": "QUESTION_GEN_product_to_formula",
    "application_to_formula": "QUESTION_GEN_application_to_formula",
    "formula_optimize": "QUESTION_GEN_formula_optimize",
    "ingredient_to_formula_and_others": "QUESTION_GEN_ingredient_to_formula_and_others",
    "ingredient_interchange": "QUESTION_GEN_ingredient_interchange",
}
