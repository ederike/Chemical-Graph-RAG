# Formatting conventions

## Never use full-width corner brackets

Do **not** use Chinese full-width corner brackets `【` `】` in:

- pretty-print / terminal output
- agent multi-hop process logs (`rag_raw_answer`, `pretty=True`)
- retrieval context labels fed to the LLM
- benchmark prompts and judge document blocks
- any new human-readable section headers

Prefer Markdown-style headings and plain labels instead:

```text
## Final Answer
### Step 1  ·  deps: none
----- Material 1 | source: foo.pdf -----
### Head
### Index block 1
```

ASCII separators (`===`, `---`) and half-width brackets `[]` / parentheses `()` are fine.
English or bilingual section titles are preferred over decorative Chinese brackets.
