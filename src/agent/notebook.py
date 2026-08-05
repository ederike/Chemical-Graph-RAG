"""工作区临时步骤记事本（Markdown）。查询结束后清空内容，不删除文件。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .state import PlanStep, StepResult


class Notebook:
    """
    路径 = settings.working_path / agent.notebook_path。
    sync 写入大纲 + 执行记录；clear 只清空内容。
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text('', encoding='utf-8')

    def write(self, text: str) -> None:
        self.path.write_text(text or '', encoding='utf-8')

    def clear(self) -> None:
        self.write('')

    def read(self) -> str:
        return self.path.read_text(encoding='utf-8') if self.path.exists() else ''

    def sync(
        self,
        *,
        query: str,
        plan: List[PlanStep],
        results: Optional[Dict[str, StepResult]] = None,
        final_answer: str = '',
    ) -> None:
        self.write(self.render(query=query, plan=plan, results=results, final_answer=final_answer))

    # ── render ───────────────────────────────────────────────────────────

    def render(
        self,
        *,
        query: str,
        plan: List[PlanStep],
        results: Optional[Dict[str, StepResult]] = None,
        final_answer: str = '',
    ) -> str:
        results = results or {}
        lines = [
            '# Agent 步骤记事本', '',
            '## 原始问题', (query or '').strip() or '（空）', '',
            '## 步骤大纲',
        ]
        if not plan:
            lines.append('（尚未规划）')
        else:
            for s in plan:
                deps = s.get('depends_on') or []
                deps_s = ','.join(str(d) for d in deps) if deps else '无'
                lines.append(f"{s.get('id', '?')}. [depends_on={deps_s}] {(s.get('question') or '').strip()}")

        lines += ['', '## 执行记录']
        if not results:
            lines.append('（尚未执行）')
        else:
            for s in plan:
                sid = str(s.get('id', ''))
                r = results.get(sid)
                if not r:
                    continue
                src = r.get('sources') or []
                ans = (r.get('answer') or '').strip() or '（空）'
                lines += [
                    f'### 步骤 {sid}',
                    f'- 规划问句：{(r.get("planned_question") or "").strip()}',
                    f'- 实际查询：{(r.get("resolved_question") or "").strip()}',
                    f'- 来源：{" / ".join(src) if src else "（无）"}',
                    '- 结论：',
                ]
                lines += [f'  {al}' for al in (ans.splitlines() or ['（空）'])]
                lines.append('')

        if final_answer:
            lines += ['## 最终答案', final_answer.strip(), '']

        return '\n'.join(lines).rstrip() + '\n'
