(() => {
  const TIMEOUT_KEY = "cgr_timeout";
  const THEME_KEY = "cgr_theme";
  const SPLIT_KEY = "cgr_split";
  const MODES = {
    retrieve: {
      label: "Retrieve 仅检索",
      path: "/api/retrieve",
      body: (q) => ({ query: q }),
    },
    dual: {
      label: "Dual-path 双路问答",
      path: "/api/query",
      body: (q) => ({ query: q, mode: "dual_path" }),
    },
    agent: {
      label: "Agent 多跳规划",
      path: "/api/multihop-query",
      body: (q) => ({ query: q }),
    },
    agentic: {
      label: "Agentic 工具循环",
      path: "/api/agentic-query",
      body: (q) => ({ query: q }),
    },
  };

  const $ = (id) => document.getElementById(id);
  const gate = $("login-gate");
  const app = $("app");
  const processBody = $("process-body");
  const answerBody = $("answer-body");
  const processChip = $("process-chip");
  const answerMeta = $("answer-meta");
  const modeBtn = $("mode-btn");
  const modeMenu = $("mode-menu");
  const modeLabel = $("mode-label");
  const settingsPop = $("settings-pop");
  const timeoutRange = $("timeout-range");
  const timeoutNum = $("timeout-num");

  let mode = "agentic";
  let asking = false;
  let authToken = null;
  let authUser = "";
  let activeCtrl = null;
  let abortTimer = null;
  let abortReason = "";
  let runTimerId = null;

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }
  function clip(s, n) {
    s = String(s ?? "").trim();
    return s.length > n ? s.slice(0, n) + "…" : s;
  }

  function renderMarkdown(src) {
    src = String(src ?? "").replace(/\r\n/g, "\n");
    if (!src.trim()) return "";

    const fences = [];
    src = src.replace(/```([^\n`]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
      const token = `\u0000F${fences.length}\u0000`;
      fences.push({ code: String(code || "").replace(/\n$/, "") });
      return token;
    });

    const lines = src.split("\n");
    const out = [];
    let i = 0;
    const fenceRe = /^\u0000F(\d+)\u0000\s*$/;

    function inline(text) {
      const codes = [];
      let s = String(text ?? "");
      s = s.replace(/`([^`]+)`/g, (_, c) => {
        const t = `\u0000C${codes.length}\u0000`;
        codes.push(esc(c));
        return t;
      });
      s = esc(s);
      s = s.replace(
        /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
      );
      s = s.replace(/\*\*([\s\S]+?)\*\*/g, "<strong>$1</strong>");
      s = s.replace(/~~([\s\S]+?)~~/g, "<del>$1</del>");
      s = s.replace(/(^|[^*])\*([^*\n]+?)\*(?!\*)/g, "$1<em>$2</em>");
      s = s.replace(/\u0000C(\d+)\u0000/g, (_, n) => `<code>${codes[Number(n)]}</code>`);
      return s;
    }

    function splitCells(row) {
      let r = String(row).trim();
      if (r.startsWith("|")) r = r.slice(1);
      if (r.endsWith("|")) r = r.slice(0, -1);
      return r.split("|").map((c) => c.trim());
    }
    function isSepRow(line) {
      if (!line || !line.includes("-")) return false;
      const cells = splitCells(line);
      return cells.length > 0 && cells.every((c) => /^:?-{3,}:?$/.test(c.replace(/\s/g, "")));
    }
    function listMatch(line) {
      const m = /^(\s*)([-*+]|\d+\.)\s+(.*)$/.exec(line);
      if (!m) return null;
      return {
        indent: m[1].replace(/\t/g, "    ").length,
        ordered: /\d/.test(m[2]),
        text: m[3],
      };
    }

    function parseList() {
      const items = [];
      while (i < lines.length) {
        const blank = lines[i].trim() === "";
        if (blank) {
          let j = i + 1;
          while (j < lines.length && lines[j].trim() === "") j++;
          if (j < lines.length && (listMatch(lines[j]) || /^\s{2,}\S/.test(lines[j]))) {
            i++;
            continue;
          }
          break;
        }
        const lm = listMatch(lines[i]);
        if (lm) {
          items.push({ ...lm, extra: [] });
          i++;
          continue;
        }
        if (items.length && /^\s{2,}\S/.test(lines[i])) {
          items[items.length - 1].extra.push(lines[i].trim());
          i++;
          continue;
        }
        break;
      }
      function renderItems(from, minIndent) {
        if (from >= items.length) return { html: "", next: from };
        const ordered = items[from].ordered;
        let html = ordered ? "<ol>" : "<ul>";
        let k = from;
        while (
          k < items.length &&
          items[k].indent === minIndent &&
          items[k].ordered === ordered
        ) {
          let body = inline(items[k].text);
          if (items[k].extra.length) body += " " + inline(items[k].extra.join(" "));
          k++;
          if (k < items.length && items[k].indent > minIndent) {
            const nested = renderItems(k, items[k].indent);
            body += nested.html;
            k = nested.next;
          }
          html += `<li>${body}</li>`;
        }
        html += ordered ? "</ol>" : "</ul>";
        if (k < items.length && items[k].indent === minIndent) {
          const more = renderItems(k, minIndent);
          html += more.html;
          k = more.next;
        }
        return { html, next: k };
      }
      if (!items.length) return "";
      const min = Math.min(...items.map((x) => x.indent));
      return renderItems(0, min).html;
    }

    while (i < lines.length) {
      const line = lines[i];
      if (line.trim() === "") {
        i++;
        continue;
      }
      const fm = fenceRe.exec(line.trim());
      if (fm) {
        out.push(`<pre><code>${esc(fences[Number(fm[1])].code)}</code></pre>`);
        i++;
        continue;
      }
      const hm = /^(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
      if (hm) {
        const lv = Math.min(4, hm[1].length);
        out.push(`<h${lv}>${inline(hm[2])}</h${lv}>`);
        i++;
        continue;
      }
      if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
        out.push("<hr>");
        i++;
        continue;
      }
      if (line.includes("|") && i + 1 < lines.length && isSepRow(lines[i + 1])) {
        const heads = splitCells(line);
        i += 2;
        const rows = [];
        while (
          i < lines.length &&
          lines[i].includes("|") &&
          lines[i].trim() &&
          !listMatch(lines[i])
        ) {
          rows.push(splitCells(lines[i]));
          i++;
        }
        let table = '<div class="table-wrap"><table><thead><tr>';
        heads.forEach((h) => {
          table += `<th>${inline(h)}</th>`;
        });
        table += "</tr></thead><tbody>";
        rows.forEach((r) => {
          table += "<tr>";
          for (let c = 0; c < heads.length; c++) table += `<td>${inline(r[c] || "")}</td>`;
          table += "</tr>";
        });
        table += "</tbody></table></div>";
        out.push(table);
        continue;
      }
      if (/^\s*>/.test(line)) {
        const qs = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) {
          qs.push(lines[i].replace(/^\s*>\s?/, ""));
          i++;
        }
        out.push(`<blockquote>${inline(qs.join(" "))}</blockquote>`);
        continue;
      }
      if (listMatch(line)) {
        out.push(parseList());
        continue;
      }
      const paras = [];
      while (i < lines.length) {
        const L = lines[i];
        if (L.trim() === "") break;
        if (fenceRe.test(L.trim())) break;
        if (/^(#{1,6})\s+/.test(L)) break;
        if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(L)) break;
        if (listMatch(L)) break;
        if (L.includes("|") && i + 1 < lines.length && isSepRow(lines[i + 1])) break;
        if (/^\s*>/.test(L)) break;
        paras.push(L);
        i++;
      }
      out.push(`<p>${inline(paras.join("\n")).replace(/\n/g, "<br>")}</p>`);
    }
    return out.join("");
  }
  function authHeader() {
    return authToken ? { Authorization: "Basic " + authToken } : {};
  }
  function getTimeout() {
    const n = Number(localStorage.getItem(TIMEOUT_KEY) || 300);
    return Math.min(900, Math.max(30, n || 300));
  }
  function setTimeoutUi(v) {
    v = Math.min(900, Math.max(30, Number(v) || 300));
    timeoutRange.value = String(v);
    timeoutNum.value = String(v);
    localStorage.setItem(TIMEOUT_KEY, String(v));
  }

  function getTheme() {
    return localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark";
  }
  function applyTheme(theme) {
    theme = theme === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
    const darkBtn = $("theme-dark");
    const lightBtn = $("theme-light");
    if (darkBtn && lightBtn) {
      darkBtn.classList.toggle("on", theme === "dark");
      lightBtn.classList.toggle("on", theme === "light");
    }
    const gateTheme = $("gate-theme");
    if (gateTheme) {
      gateTheme.textContent = theme === "light" ? "切换夜间版" : "切换白天版";
    }
  }

  function showApp() {
    document.body.dataset.view = "app";
    document.body.scrollTop = 0;
    document.documentElement.scrollTop = 0;
    const who = $("who-label");
    if (who) who.textContent = authUser || "已验证";
    closeSettings();
  }
  function showGate(msg) {
    stopQuery("logout");
    authToken = null;
    authUser = "";
    document.body.dataset.view = "login";
    closeSettings();
    const err = $("login-error");
    if (msg) {
      err.hidden = false;
      err.textContent = msg;
    } else {
      err.hidden = true;
    }
  }
  function closeSettings() {
    if (settingsPop) {
      settingsPop.hidden = true;
      $("btn-settings").setAttribute("aria-expanded", "false");
    }
  }
  function setAsking(on) {
    asking = on;
    const btn = $("ask-btn");
    if (!btn) return;
    if (on) {
      btn.textContent = "终止";
      btn.classList.add("btn-stop");
      btn.type = "button";
    } else {
      btn.textContent = "检索";
      btn.classList.remove("btn-stop");
      btn.type = "submit";
    }
  }
  function stopQuery(reason) {
    abortReason = reason || "stop";
    stopTimer();
    if (abortTimer) {
      clearTimeout(abortTimer);
      abortTimer = null;
    }
    if (activeCtrl) {
      try { activeCtrl.abort(); } catch (_) {}
    }
  }

  async function probe() {
    if (!authToken) return false;
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 8000);
    try {
      const res = await fetch("/api/health", { headers: authHeader(), signal: ctrl.signal });
      return res.ok;
    } catch {
      return false;
    } finally {
      clearTimeout(t);
    }
  }

  async function apiPost(path, body, timeoutMs) {
    activeCtrl = new AbortController();
    abortReason = "";
    abortTimer = setTimeout(() => {
      abortReason = "timeout";
      if (activeCtrl) activeCtrl.abort();
    }, timeoutMs);
    try {
      const res = await fetch(path, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeader(),
        },
        body: JSON.stringify(body),
        signal: activeCtrl.signal,
      });
      if (res.status === 401) {
        showGate("登录已失效，请重新输入密钥。");
        throw new Error("unauthorized");
      }
      let data = {};
      try { data = await res.json(); } catch { data = {}; }
      if (!res.ok) {
        const detail = data.detail;
        const msg = typeof detail === "string"
          ? detail
          : (detail && JSON.stringify(detail)) || res.statusText;
        throw new Error(msg || "请求失败");
      }
      return data;
    } catch (e) {
      if (e.name === "AbortError") {
        if (abortReason === "timeout") throw new Error("已超过设定的最长等待时间");
        if (abortReason === "logout") throw new Error("unauthorized");
        throw new Error("已终止本次查询");
      }
      throw e;
    } finally {
      if (abortTimer) {
        clearTimeout(abortTimer);
        abortTimer = null;
      }
      activeCtrl = null;
    }
  }

  function sourcesHtml(list) {
    const arr = Array.isArray(list) ? list.filter(Boolean) : [];
    if (!arr.length) return "";
    return `<div class="sources">${arr.slice(0, 12).map((s) => `<span class="src">${esc(s)}</span>`).join("")}</div>`;
  }

  function renderDual(data) {
    const ans = data.answer || "";
    processBody.innerHTML = `
      <div class="trace">
        <article class="card">
          <div class="card-kicker">Dual-path · 结果预览</div>
          <div class="preview">${esc(clip(ans, 480)) || "（无预览）"}</div>
          ${sourcesHtml(data.retrieval_sources)}
        </article>
      </div>`;
    fillAnswer(data);
  }

  function renderAgent(data) {
    const plan = Array.isArray(data.plan) ? data.plan : [];
    const steps = Array.isArray(data.steps) ? data.steps : [];
    const planHtml = plan.length
      ? `<div class="plan-row">${plan.map((p) => `
          <div class="plan-node">
            <b>Step ${esc(p.id)} · ${esc(p.kind || "retrieve")}</b>
            ${esc(clip(p.question || "", 80))}
          </div>`).join("")}</div>`
      : `<p class="muted">没有规划步骤。</p>`;
    const stepHtml = steps.map((s, i) => {
      const q = s.resolved_question || s.planned_question || s.question || "";
      return `
        <article class="card">
          <div class="card-kicker">子步骤 ${esc(s.id || i + 1)}</div>
          <h3>${esc(clip(q, 120)) || "（无问题）"}</h3>
          <div class="preview">${esc(clip(s.answer || "", 360)) || "（尚无回答）"}</div>
          ${sourcesHtml(s.sources)}
        </article>`;
    }).join("");
    processBody.innerHTML = `<div class="trace">
      <article class="card"><div class="card-kicker">规划图</div>${planHtml}</article>
      ${stepHtml}
    </div>`;
    fillAnswer(data);
  }

  function renderAgentic(data) {
    const turns = Array.isArray(data.turns) ? data.turns : [];
    if (!turns.length) {
      processBody.innerHTML = `<div class="trace"><article class="card"><div class="card-kicker">Agentic</div><p class="muted">没有回合记录。</p></article></div>`;
      fillAnswer(data);
      return;
    }
    processBody.innerHTML = `<div class="trace">${turns.map((t) => {
      const calls = Array.isArray(t.calls) ? t.calls : [];
      const callHtml = calls.map((c) => {
        const args = c.arguments && typeof c.arguments === "object"
          ? JSON.stringify(c.arguments, null, 0)
          : String(c.arguments || "");
        return `<p class="mono">▸ ${esc(c.name || "tool")} ${esc(clip(args, 180))}
${esc(clip(c.result || "", 280))}</p>`;
      }).join("");
      return `
        <article class="card">
          <div class="card-kicker">Turn ${esc(t.turn ?? "")} · ${esc(t.kind || (t.forced ? "forced" : "act"))}</div>
          ${t.thought ? `<h3>${esc(clip(t.thought, 140))}</h3>` : ""}
          ${t.thought ? `<div class="preview">${esc(clip(t.thought, 500))}</div>` : ""}
          ${callHtml || ""}
          ${t.answer ? `<div class="preview">${esc(clip(t.answer, 280))}</div>` : ""}
        </article>`;
    }).join("")}</div>`;
    fillAnswer(data);
  }

  function renderRetrieve(data) {
    const items = Array.isArray(data.items) ? data.items : [];
    processBody.innerHTML = `<div class="trace">${
      items.length
        ? items.map((it, i) => `
          <article class="card">
            <div class="card-kicker">命中 ${i + 1} · score ${esc(it.score ?? "—")}</div>
            <h3>${esc(it.source || it.node_name || "未命名来源")}</h3>
            <div class="preview">${esc(clip(it.content || "", 360))}</div>
          </article>`).join("")
        : `<article class="card"><p class="muted">没有命中片段。</p></article>`
    }</div>`;
    answerBody.innerHTML = `
      <div class="answer-text">本模式不调用生成模型，只返回检索证据。共命中 ${items.length} 条，详见上方过程栏。</div>
      ${data.retrieve_timing ? `<p class="answer-meta-line">retrieve_timing 已返回</p>` : ""}`;
    answerMeta.textContent = `${items.length} 条`;
  }

  function fillAnswer(data) {
    const ok = Number(data.status) === 1;
    const ans = data.answer || "（空回答）";
    answerBody.innerHTML = `
      <div class="answer-md">${renderMarkdown(ans)}</div>
      ${sourcesHtml(data.retrieval_sources)}
      <p class="answer-meta-line">${
        [
          data.latency_s != null ? `总耗时 ${Number(data.latency_s).toFixed(2)}s` : "",
          data.retrieve_latency_s != null ? `检索 ${Number(data.retrieve_latency_s).toFixed(2)}s` : "",
          data.usage_total_tokens != null ? `tokens ${data.usage_total_tokens}` : "",
        ].filter(Boolean).join(" · ")
      }</p>`;
    answerMeta.textContent = ok ? "完成" : "未成功";
    if (!ok) processChip.classList.add("err");
  }

  function startTimer() {
    const el = $("run-timer");
    const ta = $("query-input");
    if (!el) return;
    const t0 = performance.now();
    el.hidden = false;
    if (ta) ta.classList.add("has-timer");
    const tick = () => {
      el.textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s";
    };
    tick();
    if (runTimerId) clearInterval(runTimerId);
    runTimerId = setInterval(tick, 100);
  }
  function stopTimer() {
    if (runTimerId) {
      clearInterval(runTimerId);
      runTimerId = null;
    }
  }

  function waitingUi() {
    processChip.textContent = "进行中";
    processChip.className = "chip busy";
    answerMeta.textContent = "";
    processBody.innerHTML = `<div class="trace" id="live-trace"></div>`;
    answerBody.innerHTML = `<div class="empty"><p>正在生成最终回答…</p></div>`;
    startTimer();
  }

  function appendStep(ev) {
    let box = $("live-trace");
    if (!box) {
      processBody.innerHTML = `<div class="trace" id="live-trace"></div>`;
      box = $("live-trace");
    }
    const art = document.createElement("article");
    art.className = "card live";
    const title = ev.title || ev.stage || "步骤";
    const preview = ev.preview || "";
    art.innerHTML =
      `<div class="card-kicker">${esc(ev.stage || "step")}</div>` +
      `<h3>${esc(title)}</h3>` +
      (preview ? `<div class="preview">${esc(preview)}</div>` : "");
    box.appendChild(art);
    processBody.scrollTop = processBody.scrollHeight;
  }

  async function readSSE(res, onEvent) {
    if (!res.body || !res.body.getReader) {
      throw new Error("浏览器不支持流式读取");
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          try {
            onEvent(JSON.parse(line.slice(6)));
          } catch (_) {}
        }
      }
    }
  }

  async function streamQuery(q, timeoutMs) {
    activeCtrl = new AbortController();
    abortReason = "";
    abortTimer = setTimeout(() => {
      abortReason = "timeout";
      if (activeCtrl) activeCtrl.abort();
    }, timeoutMs);
    try {
      const res = await fetch("/api/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          ...authHeader(),
        },
        body: JSON.stringify({ query: q, mode }),
        signal: activeCtrl.signal,
      });
      if (res.status === 401) {
        showGate("登录已失效，请重新输入密钥。");
        throw new Error("unauthorized");
      }
      if (res.status === 404) {
        const spec = MODES[mode];
        return await apiPost(spec.path, spec.body(q), timeoutMs);
      }
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const err = await res.json();
          detail = err.detail || detail;
        } catch (_) {}
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      let finalData = null;
      let streamErr = null;
      await readSSE(res, (ev) => {
        if (!ev || typeof ev !== "object") return;
        if (ev.type === "step") appendStep(ev);
        else if (ev.type === "done") finalData = ev.data;
        else if (ev.type === "error") streamErr = ev.message || "stream error";
      });
      if (streamErr) throw new Error(streamErr);
      if (!finalData) throw new Error("流结束但没有最终结果");
      return finalData;
    } catch (e) {
      if (e.name === "AbortError") {
        if (abortReason === "timeout") throw new Error("已超过设定的最长等待时间");
        if (abortReason === "logout") throw new Error("unauthorized");
        throw new Error("已终止本次查询");
      }
      throw e;
    } finally {
      if (abortTimer) {
        clearTimeout(abortTimer);
        abortTimer = null;
      }
      activeCtrl = null;
    }
  }

  function pickMode(next) {
    if (!MODES[next]) return;
    mode = next;
    modeLabel.textContent = MODES[next].label;
    modeMenu.querySelectorAll("li").forEach((li) => {
      li.classList.toggle("active", li.dataset.mode === next);
    });
    modeMenu.hidden = true;
    modeBtn.setAttribute("aria-expanded", "false");
  }

  $("login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const user = $("login-user").value.trim();
    const pass = $("login-pass").value;
    const btn = $("login-btn");
    btn.disabled = true;
    authUser = user;
    authToken = btoa(unescape(encodeURIComponent(`${user}:${pass}`)));
    const ok = await probe();
    btn.disabled = false;
    if (ok) {
      $("login-pass").value = "";
      showApp();
    } else {
      authToken = null;
      authUser = "";
      showGate("用户名或密钥不正确。");
    }
  });

  $("btn-logout").addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    showGate("");
  });
  $("btn-settings").addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const open = settingsPop.hidden;
    settingsPop.hidden = !open;
    $("btn-settings").setAttribute("aria-expanded", open ? "true" : "false");
  });
  $("theme-switch").addEventListener("click", (e) => {
    e.stopPropagation();
    const btn = e.target.closest("button[data-theme]");
    if (btn) applyTheme(btn.dataset.theme);
  });
  $("gate-theme").addEventListener("click", () => {
    applyTheme(getTheme() === "light" ? "dark" : "light");
  });
  document.addEventListener("click", (e) => {
    const wrap = e.target.closest(".settings-wrap");
    if (!wrap) closeSettings();
    if (!modeMenu.hidden && !modeBtn.contains(e.target) && !modeMenu.contains(e.target)) {
      modeMenu.hidden = true;
      modeBtn.setAttribute("aria-expanded", "false");
    }
  });
  timeoutRange.addEventListener("input", () => setTimeoutUi(timeoutRange.value));
  timeoutNum.addEventListener("change", () => setTimeoutUi(timeoutNum.value));

  modeBtn.addEventListener("click", () => {
    const open = modeMenu.hidden;
    modeMenu.hidden = !open;
    modeBtn.setAttribute("aria-expanded", open ? "true" : "false");
  });
  modeMenu.addEventListener("click", (e) => {
    const li = e.target.closest("li");
    if (li) pickMode(li.dataset.mode);
  });

  $("ask-btn").addEventListener("click", (e) => {
    if (!asking) return;
    e.preventDefault();
    stopQuery("stop");
  });
  $("query-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (asking) return;
    const q = $("query-input").value.trim();
    if (!q) return;
    $("query-input").value = "";
    setAsking(true);
    waitingUi();
    try {
      const data = await streamQuery(q, getTimeout() * 1000);
      processChip.textContent = "完成";
      processChip.className = "chip";
      if (mode === "retrieve") {
        const items = Array.isArray(data.items) ? data.items : [];
        items.slice(0, 8).forEach((it, i) => appendStep({
          stage: "hit",
          title: it.source || `命中 ${i + 1}`,
          preview: it.content || "",
        }));
        answerBody.innerHTML = `<div class="answer-text">本模式不调用生成模型，只返回检索证据。共命中 ${items.length} 条，详见上方过程栏。</div>`;
        answerMeta.textContent = `${items.length} 条`;
      } else if ($("live-trace") && $("live-trace").children.length) {
        fillAnswer(data);
      } else if (mode === "dual") renderDual(data);
      else if (mode === "agent") renderAgent(data);
      else if (mode === "agentic") renderAgentic(data);
      else renderRetrieve(data);
      answerBody.scrollTop = 0;
    } catch (err) {
      if (err.message === "unauthorized") return;
      const stopped = err.message === "已终止本次查询";
      processChip.textContent = stopped ? "已终止" : "失败";
      processChip.className = stopped ? "chip" : "chip err";
      appendStep({
        stage: stopped ? "stop" : "error",
        title: stopped ? "已停止" : "错误",
        preview: err.message,
      });
      answerBody.innerHTML = `<div class="empty"><p>${stopped ? "本次查询已终止。" : "没有最终回答。"}</p></div>`;
    } finally {
      stopTimer();
      setAsking(false);
    }
  });

  $("query-input").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      $("query-form").requestSubmit();
    }
  });

  function isNarrowSplit() {
    return window.matchMedia("(max-width: 720px)").matches;
  }
  function applySplit(pct) {
    pct = Math.min(78, Math.max(22, Number(pct) || 52));
    const workspace = $("workspace");
    if (workspace) workspace.style.setProperty("--split", pct + "%");
    try { localStorage.setItem(SPLIT_KEY, String(Math.round(pct * 10) / 10)); } catch (_) {}
    return pct;
  }
  function initSplit() {
    const workspace = $("workspace");
    const gutter = $("split-gutter");
    let saved = 52;
    try { saved = Number(localStorage.getItem(SPLIT_KEY) || 52); } catch (_) {}
    applySplit(saved);
    if (!workspace || !gutter) return;

    const syncOri = () => {
      gutter.setAttribute("aria-orientation", isNarrowSplit() ? "horizontal" : "vertical");
    };
    syncOri();
    window.addEventListener("resize", syncOri);

    let dragging = false;
    const posToPct = (x, y) => {
      const r = workspace.getBoundingClientRect();
      if (r.width < 1 || r.height < 1) return 52;
      return isNarrowSplit()
        ? ((y - r.top) / r.height) * 100
        : ((x - r.left) / r.width) * 100;
    };
    const onMove = (ev) => {
      if (!dragging) return;
      const t = ev.touches ? ev.touches[0] : ev;
      applySplit(posToPct(t.clientX, t.clientY));
      ev.preventDefault();
    };
    const onUp = () => {
      dragging = false;
      document.body.classList.remove("is-resizing", "is-resizing-row");
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("touchend", onUp);
    };
    const startDrag = (ev) => {
      dragging = true;
      document.body.classList.add(isNarrowSplit() ? "is-resizing-row" : "is-resizing");
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
      window.addEventListener("touchmove", onMove, { passive: false });
      window.addEventListener("touchend", onUp);
      ev.preventDefault();
    };
    gutter.addEventListener("mousedown", startDrag);
    gutter.addEventListener("touchstart", startDrag, { passive: false });
    gutter.addEventListener("dblclick", () => applySplit(52));
    gutter.addEventListener("keydown", (ev) => {
      const cur = Number(String(workspace.style.getPropertyValue("--split") || "52").replace("%", "")) || 52;
      const step = ev.shiftKey ? 8 : 3;
      if (ev.key === "ArrowLeft" || ev.key === "ArrowUp") { applySplit(cur - step); ev.preventDefault(); }
      else if (ev.key === "ArrowRight" || ev.key === "ArrowDown") { applySplit(cur + step); ev.preventDefault(); }
      else if (ev.key === "Home") { applySplit(22); ev.preventDefault(); }
      else if (ev.key === "End") { applySplit(78); ev.preventDefault(); }
      else if (ev.key === "Enter") { applySplit(52); ev.preventDefault(); }
    });
  }

  setTimeoutUi(getTimeout());
  applyTheme(getTheme());
  pickMode("agentic");
  initSplit();
  document.body.dataset.view = "login";
})();
