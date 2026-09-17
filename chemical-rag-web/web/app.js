(() => {
  const TIMEOUT_KEY = "cgr_timeout";
  const THEME_KEY = "cgr_theme";
  const SPLIT_KEY = "cgr_split";
  const TOKEN_KEY = "cgr_token";
  const USER_KEY = "cgr_user";
  const SIDEBAR_KEY = "cgr_sidebar";
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
  let conversationId = null;
  let conversations = [];
  let currentTurns = [];
  let liveSteps = [];
  let lastQuery = "";
  let paintAnswer = true;
  let viewingTurnId = null;
  let maxContext = 0;
  let contextReserve = 4096;
  let contextModel = "";

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
    return authToken ? { Authorization: "Bearer " + authToken } : {};
  }
  function persistAuth() {
    try {
      if (authToken) {
        localStorage.setItem(TOKEN_KEY, authToken);
        localStorage.setItem(USER_KEY, authUser || "");
      } else {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
      }
    } catch (_) {}
  }
  async function apiJson(method, path, body, opts) {
    const res = await fetch(path, {
      method,
      headers: {
        ...(body != null ? { "Content-Type": "application/json" } : {}),
        ...authHeader(),
      },
      body: body != null ? JSON.stringify(body) : undefined,
    });
    let data = {};
    try { data = await res.json(); } catch { data = {}; }
    if (res.status === 401) {
      const errBody = data && data.detail;
      const detail = typeof errBody === "string" && errBody ? errBody : "unauthorized";
      if (!opts || !opts.silent) showGate("登录已失效，请重新输入密钥。");
      throw new Error(detail);
    }
    if (!res.ok) {
      const detail = data.detail;
      const msg = typeof detail === "string"
        ? detail
        : (detail && JSON.stringify(detail)) || res.statusText;
      throw new Error(msg || "请求失败");
    }
    return data;
  }
  function groupLabel(iso) {
    const t = Date.parse(iso);
    if (!Number.isFinite(t)) return "更早";
    const d = new Date(t);
    const now = new Date();
    const start = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const day = new Date(d.getFullYear(), d.getMonth(), d.getDate());
    const diff = (start - day) / 86400000;
    if (diff <= 0) return "今天";
    if (diff === 1) return "昨天";
    if (diff < 7) return "7 天内";
    if (diff < 30) return "30 天内";
    return "更早";
  }
  function renderSessionList() {
    const nav = $("session-nav");
    if (!nav) return;
    const q = (($("session-search") && $("session-search").value) || "").trim().toLowerCase();
    const items = conversations.filter((c) => {
      if (!q) return true;
      return (c.title || "").toLowerCase().includes(q) || (c.last_query || "").toLowerCase().includes(q);
    });
    if (!items.length) {
      nav.innerHTML = `<div class="session-empty">${
        conversations.length ? "没有匹配的对话。" : "还没有对话。点「新对话」或直接提问。"
      }</div>`;
      return;
    }
    let html = "";
    let lastG = "";
    items.forEach((c) => {
      const g = groupLabel(c.updated_at);
      if (g !== lastG) {
        html += `<div class="session-group">${esc(g)}</div>`;
        lastG = g;
      }
      const on = c.id === conversationId ? " on" : "";
      html += `<div class="session-item${on}" data-id="${esc(c.id)}" title="${esc(c.title || "")}">
        <span class="session-name">${esc(c.title || "新对话")}</span>
        <span class="session-actions">
          <button type="button" data-act="rename" title="重命名">改</button>
          <button type="button" data-act="delete" class="danger" title="删除">删</button>
        </span>
      </div>`;
    });
    nav.innerHTML = html;
  }
  async function refreshSessions() {
    const data = await apiJson("GET", "/api/conversations");
    conversations = Array.isArray(data.items) ? data.items : [];
    renderSessionList();
  }
  function emptyProcess() {
    processChip.textContent = "待命";
    processChip.className = "chip";
    processBody.innerHTML = `<div class="empty">
      <p>过程轨迹会出现在这里。</p>
      <p class="muted">Agentic 按轮次展开工具调用；Agent 展示规划与子步骤；Dual-path / 仅检索给出预览。</p>
    </div>`;
  }
  function emptyAnswer() {
    answerMeta.textContent = "";
    answerBody.innerHTML = `<div class="empty"><p>模型的最终结论会写在这一栏。左侧可切换本账号的历史对话。</p></div>`;
  }
  function startDraft() {
    conversationId = null;
    currentTurns = [];
    liveSteps = [];
    viewingTurnId = null;
    const title = $("session-title");
    if (title) title.textContent = "新对话";
    emptyProcess();
    emptyAnswer();
    renderSessionList();
    updateCtxBar();
  }
  function renderThread(turns, activeId) {
    currentTurns = turns || [];
    if (!currentTurns.length) {
      emptyAnswer();
      return;
    }
    const aid = activeId || currentTurns[currentTurns.length - 1].id;
    answerBody.innerHTML = currentTurns.map((t) => {
      const ans = t.answer || (t.result && t.result.answer) || "";
      const retrieve = t.mode === "retrieve";
      const body = retrieve
        ? `<div class="answer-text">${esc(ans || "本模式不调用生成模型，只返回检索证据。")}</div>`
        : `<div class="answer-md">${renderMarkdown(ans)}</div>`;
      return `<article class="turn-block${t.id === aid ? " on" : ""}" data-turn="${t.id}">
        <div class="turn-kicker">第 ${esc(t.turn_index)} 问 · ${esc(t.mode)}</div>
        <div class="turn-q">${esc(t.query)}</div>
        ${body}
        ${sourcesHtml(t.sources || (t.result && t.result.retrieval_sources) || [])}
      </article>`;
    }).join("");
    answerBody.scrollTop = answerBody.scrollHeight;
  }
  function showTurnProcess(turn) {
    if (!turn) {
      viewingTurnId = null;
      emptyProcess();
      return;
    }
    viewingTurnId = turn.id;
    const data = turn.result && typeof turn.result === "object" ? turn.result : {};
    const steps = Array.isArray(turn.stream_steps) ? turn.stream_steps : [];
    processChip.textContent = Number(turn.status) === 1 || Number(data.status) === 1 ? "完成" : "记录";
    processChip.className = "chip";
    answerMeta.textContent = turn.mode || "";
    paintAnswer = false;
    try {
      if (steps.length) {
        processBody.innerHTML = `<div class="trace" id="live-trace"></div>`;
        steps.forEach((ev) => appendStep(ev, { noScroll: true }));
        return;
      }
      if (turn.mode === "retrieve") renderRetrieve(Object.assign({ items: [] }, data));
      else if (turn.mode === "dual") renderDual(data);
      else if (turn.mode === "agent") renderAgent(data);
      else renderAgentic(data);
    } finally {
      paintAnswer = true;
    }
  }
  async function loadConversation(id) {
    const full = await apiJson("GET", `/api/conversations/${id}`);
    conversationId = full.id;
    currentTurns = Array.isArray(full.turns) ? full.turns : [];
    const title = $("session-title");
    if (title) title.textContent = full.title || "对话";
    const last = currentTurns[currentTurns.length - 1];
    renderThread(currentTurns, last && last.id);
    if (last) showTurnProcess(last);
    else emptyProcess();
    renderSessionList();
    closeMobileSidebar();
    updateCtxBar();
  }
  async function persistTurn(query, data) {
    if (!conversationId) {
      const created = await apiJson("POST", "/api/conversations", { title: clip(query, 40) });
      conversationId = created.id;
    }
    const retrieveNote = data && Array.isArray(data.items)
      ? `本模式不调用生成模型，只返回检索证据。共命中 ${data.items.length} 条。`
      : "";
    await apiJson("POST", `/api/conversations/${conversationId}/turns`, {
      query,
      mode,
      answer: (data && data.answer) || retrieveNote,
      status: data && data.status != null ? data.status : 1,
      latency_s: data && data.latency_s != null ? data.latency_s : null,
      sources: (data && data.retrieval_sources) || [],
      result: data || {},
      stream_steps: liveSteps.slice(),
    });
    await refreshSessions();
    const full = await apiJson("GET", `/api/conversations/${conversationId}`);
    currentTurns = Array.isArray(full.turns) ? full.turns : [];
    const title = $("session-title");
    if (title) title.textContent = full.title || "对话";
    renderThread(currentTurns, currentTurns.length ? currentTurns[currentTurns.length - 1].id : null);
    updateCtxBar();
  }
  function setSidebarCollapsed(on) {
    document.documentElement.classList.toggle("sidebar-collapsed", !!on);
    document.documentElement.classList.remove("sidebar-open");
    try { localStorage.setItem(SIDEBAR_KEY, on ? "collapsed" : "open"); } catch (_) {}
    const scrim = $("sidebar-scrim");
    if (scrim) scrim.hidden = true;
  }
  function openMobileSidebar() {
    document.documentElement.classList.add("sidebar-open");
    const scrim = $("sidebar-scrim");
    if (scrim) scrim.hidden = false;
  }
  function closeMobileSidebar() {
    document.documentElement.classList.remove("sidebar-open");
    const scrim = $("sidebar-scrim");
    if (scrim) scrim.hidden = true;
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
    startDraft();
    loadContextInfo();
    refreshSessions().catch(() => {});
  }
  function showGate(msg) {
    stopQuery("logout");
    const prev = authToken;
    authToken = null;
    authUser = "";
    persistAuth();
    conversationId = null;
    conversations = [];
    currentTurns = [];
    document.body.dataset.view = "login";
    closeSettings();
    closeMobileSidebar();
    if (prev) {
      fetch("/api/auth/logout", {
        method: "POST",
        headers: { Authorization: "Bearer " + prev },
      }).catch(() => {});
    }
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
    try {
      const data = await apiJson("GET", "/api/auth/me", null, { silent: true });
      if (data && data.username) authUser = data.username;
      return true;
    } catch {
      return false;
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

  function paintRetrieveAnswer(data) {
    const items = Array.isArray(data.items) ? data.items : [];
    if (!paintAnswer) return;
    answerBody.innerHTML = `
      <div class="answer-text">本模式不调用生成模型，只返回检索证据。共命中 ${items.length} 条，详见上方过程栏。</div>
      ${data.retrieve_timing ? `<p class="answer-meta-line">retrieve_timing 已返回</p>` : ""}`;
    answerMeta.textContent = `${items.length} 条`;
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
    paintRetrieveAnswer(data);
  }

  function fillAnswer(data) {
    const ok = Number(data.status) === 1;
    answerMeta.textContent = ok ? "完成" : "未成功";
    if (!ok) processChip.classList.add("err");
    if (!paintAnswer) return;
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
    liveSteps = [];
    processChip.textContent = "进行中";
    processChip.className = "chip busy";
    answerMeta.textContent = "";
    processBody.innerHTML = `<div class="trace" id="live-trace"></div>`;
    const pending = document.createElement("article");
    pending.id = "pending-turn";
    pending.className = "turn-block";
    pending.innerHTML = `<div class="turn-kicker">进行中</div><div class="empty"><p>正在生成最终回答…</p></div>`;
    if (!answerBody.querySelector(".turn-block")) answerBody.innerHTML = "";
    const old = $("pending-turn");
    if (old) old.remove();
    answerBody.appendChild(pending);
    answerBody.scrollTop = answerBody.scrollHeight;
    startTimer();
  }

  function appendStep(ev, opts) {
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
    if (!opts || !opts.noScroll) {
      processBody.scrollTop = processBody.scrollHeight;
    }
    if (asking) {
      liveSteps.push({
        stage: ev.stage || "step",
        title: ev.title || ev.stage || "步骤",
        preview: ev.preview || "",
      });
    }
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

  function historyPayload() {
    return (currentTurns || [])
      .filter((t) => t && String(t.query || "").trim())
      .map((t) => {
        const res = t.result && typeof t.result === "object" ? t.result : {};
        const pt = Number(res.last_prompt_tokens || res.usage_prompt_tokens || 0) || 0;
        const row = { query: String(t.query), answer: String(t.answer || "") };
        if (pt > 0) {
          row.last_prompt_tokens = pt;
          row.usage_prompt_tokens = pt;
        }
        return row;
      });
  }
  function fmtTokens(n) {
    n = Math.max(0, Math.round(Number(n) || 0));
    if (n >= 102400) return (n / 1024).toFixed(0) + "k";
    if (n >= 1024) return (n / 1024).toFixed(1).replace(/\.0$/, "") + "k";
    return String(n);
  }
  function estimateLocalTokens(turns, extraQ) {
    const list = turns || [];
    let chars = 0;
    list.forEach((t) => {
      chars += String(t.query || "").length + String(t.answer || "").length;
    });
    chars += String(extraQ || "").length;
    const overhead = list.length || extraQ ? 8000 : 0;
    const fromChars = Math.floor(chars / 2) + overhead;
    let lastPt = 0;
    let lastAns = "";
    for (let i = list.length - 1; i >= 0; i--) {
      const res = list[i].result && typeof list[i].result === "object" ? list[i].result : {};
      lastPt = Number(res.last_prompt_tokens || res.usage_prompt_tokens || 0) || 0;
      lastAns = String(list[i].answer || "");
      if (lastPt > 0) break;
    }
    if (lastPt > 0) {
      const extra = Math.floor(String(extraQ || "").length / 2) + Math.floor(lastAns.length / 2);
      return Math.max(fromChars, lastPt + extra);
    }
    return fromChars;
  }
  function updateCtxBar(extraQ) {
    const bar = $("ctx-bar");
    const fill = $("ctx-bar-fill");
    const label = $("ctx-bar-label");
    if (!bar || !fill || !label) return;
    const used = estimateLocalTokens(currentTurns, extraQ || "");
    const max = maxContext || 0;
    const limit = max > 0 ? Math.max(1, max - (contextReserve || 0)) : 0;
    const pct = max > 0 ? Math.min(100, (used / max) * 100) : 0;
    fill.style.width = (max ? pct : 0) + "%";
    bar.classList.remove("warn", "danger", "over");
    if (max && used >= limit) bar.classList.add("over");
    else if (pct >= 90) bar.classList.add("danger");
    else if (pct >= 70) bar.classList.add("warn");
    if (!max) {
      label.textContent = "上下文 —";
      return;
    }
    const model = contextModel ? ` · ${contextModel}` : "";
    label.textContent = `上下文 ${fmtTokens(used)} / ${fmtTokens(max)}${model}`;
    bar.title = `当前会话预计占用 ${used} / ${max} tokens（模型窗口 ${max}，预留 ${contextReserve}）`;
  }
  async function loadContextInfo() {
    try {
      const info = await apiJson("GET", "/api/context", null, { silent: true });
      maxContext = Number(info.max_context_tokens) || 0;
      contextReserve = Number(info.reserve_tokens) || 4096;
      contextModel = info.model || "";
    } catch (_) {}
    updateCtxBar();
  }
  function contextWouldOverflow(extraQ) {
    if (!maxContext) return false;
    const used = estimateLocalTokens(currentTurns, extraQ || "");
    const limit = Math.max(1, maxContext - (contextReserve || 0));
    return used >= limit;
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
        body: JSON.stringify({ query: q, mode, history: historyPayload() }),
        signal: activeCtrl.signal,
      });
      if (res.status === 401) {
        showGate("登录已失效，请重新输入密钥。");
        throw new Error("unauthorized");
      }
      if (res.status === 404) {
        const spec = MODES[mode];
        return await apiPost(spec.path, { ...spec.body(q), history: historyPayload() }, timeoutMs);
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
    try {
      const data = await apiJson("POST", "/api/auth/login", { username: user, password: pass }, { silent: true });
      authToken = data.token;
      authUser = data.username || user;
      persistAuth();
      $("login-pass").value = "";
      showApp();
    } catch (err) {
      authToken = null;
      authUser = "";
      persistAuth();
      showGate(err.message === "unauthorized" ? "用户名或密钥不正确。" : (err.message || "登录失败"));
    } finally {
      btn.disabled = false;
    }
  });

  $("btn-logout").addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    showGate("");
  });
  $("btn-new-chat").addEventListener("click", (e) => {
    e.preventDefault();
    startDraft();
    closeMobileSidebar();
    $("query-input").focus();
  });
  $("session-search").addEventListener("input", () => renderSessionList());
  $("session-nav").addEventListener("click", async (e) => {
    const item = e.target.closest(".session-item");
    if (!item) return;
    const id = item.dataset.id;
    const act = e.target.closest("button[data-act]");
    try {
      if (act && act.dataset.act === "rename") {
        e.preventDefault();
        const cur = conversations.find((c) => c.id === id);
        const title = window.prompt("对话标题", (cur && cur.title) || "");
        if (!title || !title.trim()) return;
        await apiJson("PATCH", `/api/conversations/${id}`, { title: title.trim() });
        await refreshSessions();
        if (conversationId === id && $("session-title")) {
          $("session-title").textContent = title.trim();
        }
        return;
      }
      if (act && act.dataset.act === "delete") {
        e.preventDefault();
        if (!window.confirm("删除这条对话？不可恢复。")) return;
        await apiJson("DELETE", `/api/conversations/${id}`);
        if (conversationId === id) startDraft();
        await refreshSessions();
        return;
      }
      if (id && id !== conversationId) await loadConversation(id);
    } catch (err) {
      if (err.message !== "unauthorized") window.alert(err.message || "操作失败");
    }
  });
  answerBody.addEventListener("click", (e) => {
    const qel = e.target.closest(".turn-q");
    if (!qel) return;
    const block = qel.closest(".turn-block");
    if (!block) return;
    const id = Number(block.dataset.turn);
    const turn = currentTurns.find((t) => t.id === id);
    if (!turn) return;
    if (viewingTurnId === turn.id) return;
    answerBody.querySelectorAll(".turn-block").forEach((el) => {
      el.classList.toggle("on", el === block);
    });
    showTurnProcess(turn);
  });
  $("btn-sidebar-collapse").addEventListener("click", () => {
    if (window.matchMedia("(max-width: 860px)").matches) closeMobileSidebar();
    else setSidebarCollapsed(true);
  });
  $("btn-sidebar-open").addEventListener("click", () => {
    if (window.matchMedia("(max-width: 860px)").matches) openMobileSidebar();
    else setSidebarCollapsed(false);
  });
  $("sidebar-scrim").addEventListener("click", () => closeMobileSidebar());
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
    if (contextWouldOverflow(q)) {
      updateCtxBar(q);
      window.alert(
        `当前对话上下文已达到回答模型上限（${fmtTokens(maxContext)} tokens，模型 ${contextModel || "LLM"}）。请点击左侧「新对话」开一个新会话再继续。`
      );
      return;
    }
    lastQuery = q;
    $("query-input").value = "";
    setAsking(true);
    waitingUi();
    updateCtxBar(q);
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
      } else if ($("live-trace") && $("live-trace").children.length) {
        /* process already streamed */
      } else if (mode === "dual") renderDual(data);
      else if (mode === "agent") renderAgent(data);
      else if (mode === "agentic") renderAgentic(data);
      else renderRetrieve(data);
      await persistTurn(q, data);
    } catch (err) {
      if (err.message === "unauthorized") return;
      if (String(err.message || "").includes("请点击「新对话」") || String(err.message || "").includes("已达到回答模型上限")) {
        updateCtxBar(q);
        window.alert(err.message);
        return;
      }
      const stopped = err.message === "已终止本次查询";
      processChip.textContent = stopped ? "已终止" : "失败";
      processChip.className = stopped ? "chip" : "chip err";
      appendStep({
        stage: stopped ? "stop" : "error",
        title: stopped ? "已停止" : "错误",
        preview: err.message,
      });
      try {
        await persistTurn(q, {
          status: 0,
          answer: stopped ? "本次查询已终止。" : (err.message || "没有最终回答。"),
        });
      } catch (_) {
        const pending = $("pending-turn");
        if (pending) pending.remove();
        if (!answerBody.querySelector(".turn-block")) {
          answerBody.innerHTML = `<div class="empty"><p>${stopped ? "本次查询已终止。" : "没有最终回答。"}</p></div>`;
        }
      }
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
  (async () => {
    try {
      authToken = localStorage.getItem(TOKEN_KEY) || null;
      authUser = localStorage.getItem(USER_KEY) || "";
    } catch (_) {}
    if (authToken && await probe()) {
      persistAuth();
      showApp();
    } else {
      authToken = null;
      authUser = "";
      persistAuth();
      document.body.dataset.view = "login";
    }
  })();
})();
