"""工作区临时步骤记事本（Markdown）。查询结束后清空内容，不删除文件。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .state import PlanStep, StepResult


class Notebook:
    """
    固定目录 = settings.working_path，文件名由 agent.notebook_path 配置。
    写入：大纲 + 逐步执行记录；clear() 只清空内容。
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text('', encoding='utf-8')

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    def write(self, text: str) -> None:
        self.path.write_text(text or '', encoding='utf-8')

    def clear(self) -> None:
        self.write('')

    def read(self) -> str:
        if not self.path.exists():
            return ''
        return self.path.read_text(encoding='utf-8')

    # ------------------------------------------------------------------
    # 结构化渲染
    # ------------------------------------------------------------------
    def render_full(
        self,
        *,
        query: str,
        plan: List[PlanStep],
        results: Optional[Dict[str, StepResult]] = None,
        final_answer: str = '',
    ) -> str:
        results = results or {}
        lines = [
            '# Agent 步骤记事本',
            '',
            '## 原始问题',
            query.strip() if query else '（空）',
            '',
            '## 步骤大纲',
        ]
        if not plan:
            lines.append('（尚未规划）')
        else:
            for step in plan:
                sid = step.get('id', '?')
                deps = step.get('depends_on') or []
                deps_s = ','.join(str(d) for d in deps) if deps else '无'
                q = (step.get('question') or '').strip()
                lines.append(f'{sid}. [depends_on={deps_s}] {q}')

        lines.extend(['', '## 执行记录'])
        if not results:
            lines.append('（尚未执行）')
        else:
            for step in plan:
                sid = str(step.get('id', ''))
                r = results.get(sid)
                if not r:
                    continue
                lines.append(f'### 步骤 {sid}')
                lines.append(f'- 规划问句：{(r.get("planned_question") or "").strip()}')
                lines.append(f'- 实际查询：{(r.get("resolved_question") or "").strip()}')
                src = r.get('sources') or []
                lines.append(f'- 来源：{" / ".join(src) if src else "（无）"}')
                lines.append(f'- 结论：')
                ans = (r.get('answer') or '').strip() or '（空）'
                for al in ans.splitlines() or ['（空）']:
                    lines.append(f'  {al}')
                lines.append('')

        if final_answer:
            lines.extend(['## 最终答案', final_answer.strip(), ''])

        return '\n'.join(lines).rstrip() + '\n'

    def sync(
        self,
        *,
        query: str,
        plan: List[PlanStep],
        results: Optional[Dict[str, StepResult]] = None,
        final_answer: str = '',
    ) -> None:
        self.write(self.render_full(
            query=query,
            plan=plan,
            results=results,
            final_answer=final_answer,
        ))
