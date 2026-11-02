const RP = (() => {
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

  // Markdown subset: paragraphs, "- " lists, **bold**, `code`, [text](url), [[node-id]] / [[node-id|text]].
  // $...$ is left untouched for KaTeX auto-render.
  function md(src, nodeLink) {
    if (src == null) return "";
    const inline = t => {
      const codes = [];
      t = esc(t).replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
      t = t.replace(/\[(lõi|diễn giải|nhận xét của AI|tự tính|nguồn khác|không có trong paper|chỉ từ abstract)\](?!\()/g, (_, tag) =>
             `<span class="badge${tag === "lõi" ? " acc" : /không có|nguồn khác/.test(tag) ? " warn" : ""}">${tag}</span>`)
           .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
           .replace(/\[\[([\w-]+)(?:\|([^\]]+))?\]\]/g, (_, id, label) => nodeLink ? nodeLink(id, label) : esc(label || id))
           .replace(/\[([^\]]+)\]\((https?:[^)\s]+|[\w./#-]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
      return t.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[i]}</code>`);
    };
    return String(src).trim().split(/\n\s*\n/).map(block => {
      const lines = block.split("\n");
      if (lines.every(l => /^\s*- /.test(l))) return "<ul>" + lines.map(l => `<li>${inline(l.replace(/^\s*- /, ""))}</li>`).join("") + "</ul>";
      return `<p>${lines.map(inline).join("<br>")}</p>`;
    }).join("");
  }

  function flow(steps) {
    return `<div class="flow">` + steps.map((s, i) =>
      (i ? `<div class="arrow">→</div>` : "") +
      `<div class="step${s.new ? " new" : ""}"><b>${esc(s.label)}</b>${s.detail ? `<span>${esc(s.detail)}</span>` : ""}</div>`
    ).join("") + `</div>`;
  }

  function bars(b) {
    const max = b.max ?? Math.max(...b.items.map(i => Math.abs(i.value)));
    return (b.title ? `<h3>${esc(b.title)}</h3>` : "") + `<div class="bars">` + b.items.map(i =>
      `<div>${esc(i.label)}</div><div class="track"><div class="fill${i.highlight ? " hi" : ""}" style="width:${max ? Math.abs(i.value) / max * 100 : 0}%"></div></div><div class="val">${esc(i.value)}${esc(b.unit || "")}</div>`
    ).join("") + `</div>` + (b.src ? `<div class="src">${esc(b.src)}</div>` : "");
  }

  function table(t, nodeLink) {
    return `<div class="tablewrap"><table><thead><tr>${t.head.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>` +
      t.rows.map(r => `<tr>${r.map(c => `<td>${md(String(c ?? ""), nodeLink).replace(/^<p>|<\/p>$/g, "")}</td>`).join("")}</tr>`).join("") +
      `</tbody></table></div>` + (t.src ? `<div class="src">${esc(t.src)}</div>` : "");
  }

  function formula(f) {
    const syms = (f.symbols || []).map(([s, m]) => `<tr><td>$${s}$</td><td>${esc(m)}</td></tr>`).join("");
    return `<div class="formula">$$${f.expr}$$</div>` +
      (f.say ? `<div class="say"><b>Đọc là:</b> ${md(f.say).replace(/^<p>|<\/p>$/g, "")}</div>` : "") +
      (syms ? `<div class="tablewrap"><table class="sym"><thead><tr><th>Ký hiệu</th><th>Nghĩa</th></tr></thead><tbody>${syms}</tbody></table></div>` : "") +
      (f.src ? `<div class="src">${esc(f.src)}</div>` : "");
  }

  // Vectors drawn as rows of numbered cells; cell tint follows sign and size so the eye sees the
  // pattern (which coordinates are large, which are zero) before reading the numbers.
  function cellColor(v, max) {
    if (typeof v !== "number" || !max || v === 0) return "var(--bg)";
    const a = Math.min(1, Math.abs(v) / max) * 0.55 + 0.08;
    return v > 0 ? `rgba(51,126,169,${a.toFixed(2)})` : `rgba(212,76,71,${a.toFixed(2)})`;
  }
  function vectors(b, nodeLink) {
    const nums = b.rows.flatMap(r => r.values || []).filter(v => typeof v === "number");
    const max = Math.max(...nums.map(Math.abs), 0);
    const item = r => r.op
      ? `<div class="vop">${esc(r.op)}</div>`
      : `<div class="vec${r.hl ? " hl" : ""}"><div class="vlabel">${md(r.label || "", nodeLink).replace(/^<p>|<\/p>$/g, "")}</div>` +
        `<div class="cells">${(r.values || []).map((v, i) => `<div class="cell" style="background:${cellColor(v, max)}" title="${esc(b.cellLabels?.[i] || "")}">${esc(v)}</div>`).join("")}</div>` +
        (r.note ? `<div class="vnote">${esc(r.note)}</div>` : "") + `</div>`;
    return (b.title ? `<h3>${esc(b.title)}</h3>` : "") + `<div class="vecs">${b.rows.map(item).join("")}</div>` +
      (b.cellLabels ? `<div class="src">Các ô từ trái sang phải: ${esc(b.cellLabels.join(", "))}</div>` : "") +
      (b.note ? `<div class="vcap">${md(b.note, nodeLink)}</div>` : "");
  }

  const PCOL = {acc: "#337ea9", warn: "#cb912f", ok: "#448361", bad: "#d44c47", mute: "#9b9a97", ink: "#37352f"};
  function plot2d(b, nodeLink) {
    const S = 280, P = 34, R = b.range || Math.max(1, ...b.vectors.flatMap(v => [Math.abs(v.x), Math.abs(v.y)])) * 1.15;
    const sx = x => P + (x + R) / (2 * R) * (S - 2 * P), sy = y => S - P - (y + R) / (2 * R) * (S - 2 * P);
    const ax = b.axes || ["chiều 1", "chiều 2"];
    let g = `<line x1="${P}" y1="${sy(0)}" x2="${S - P}" y2="${sy(0)}" class="ax"/><line x1="${sx(0)}" y1="${S - P}" x2="${sx(0)}" y2="${P}" class="ax"/>` +
      `<text x="${S - P}" y="${sy(0) + 16}" text-anchor="end" class="axl">${esc(ax[0])} →</text>` +
      `<text x="${sx(0) + 6}" y="${P - 8}" class="axl">↑ ${esc(ax[1])}</text>`;
    b.vectors.forEach((v, i) => {
      const c = PCOL[v.color] || PCOL.acc, x = sx(v.x), y = sy(v.y);
      if (v.proj) {
        g += `<line x1="${x}" y1="${y}" x2="${sx(0)}" y2="${y}" stroke="${c}" stroke-dasharray="3 3" stroke-width="1"/>` +
             `<text x="${sx(0) - 4}" y="${y + 4}" text-anchor="end" class="pl" fill="${c}">${esc(v.y)}</text>`;
      }
      g += `<defs><marker id="pa${i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="${c}"/></marker></defs>` +
        `<line x1="${sx(0)}" y1="${sy(0)}" x2="${x}" y2="${y}" stroke="${c}" stroke-width="2.2"${v.dashed ? ' stroke-dasharray="5 4"' : ""} marker-end="url(#pa${i})"/>` +
        `<text x="${x + (v.x >= 0 ? 6 : -6)}" y="${y + (v.y >= 0 ? -6 : 14)}" text-anchor="${v.x >= 0 ? "start" : "end"}" class="vl" fill="${c}">${esc(v.label)}</text>`;
    });
    return (b.title ? `<h3>${esc(b.title)}</h3>` : "") + `<div class="plot"><svg viewBox="0 0 ${S} ${S}" width="${S}" height="${S}">${g}</svg>` +
      (b.note ? `<div class="vcap">${md(b.note, nodeLink)}</div>` : "") + `</div>`;
  }

  function steps(b, nodeLink) {
    return (b.title ? `<h3>${esc(b.title)}</h3>` : "") + `<ol class="steps">${b.items.map(s =>
      `<li><div class="st">${md(s.text, nodeLink).replace(/^<p>|<\/p>$/g, "")}</div>` +
      (s.expr ? `<div class="sx"><span class="mono">${esc(s.expr)}</span>${s.result != null ? ` <span class="sr">= ${esc(s.result)}</span>` : ""}</div>` : "") + `</li>`).join("")}</ol>` +
      (b.note ? `<div class="vcap">${md(b.note, nodeLink)}</div>` : "");
  }

  // One renderer for every visual block, shared by the roadmap detail pages and the deep dive.
  function visual(b, nodeLink) {
    const strip = s => s.replace(/^<p>|<\/p>$/g, "");
    switch (b.type) {
      case "text": return md(b.md, nodeLink);
      case "heading": return `<h3>${esc(b.text)}</h3>`;
      case "flow": return (b.title ? `<h3>${esc(b.title)}</h3>` : "") + flow(b.steps) + (b.note ? `<div class="vcap">${md(b.note, nodeLink)}</div>` : "") + (b.src ? `<div class="src">${esc(b.src)}</div>` : "");
      case "vectors": return vectors(b, nodeLink);
      case "plot2d": return plot2d(b, nodeLink);
      case "steps": return steps(b, nodeLink);
      case "bars": return bars(b) + (b.note ? `<div class="vcap">${md(b.note, nodeLink)}</div>` : "");
      case "table": return (b.title ? `<h3>${esc(b.title)}</h3>` : "") + table(b, nodeLink);
      case "formula": return formula(b);
      case "quote": return `<blockquote>${md(b.text, nodeLink)}${b.src ? `<div class="src">${esc(b.src)}</div>` : ""}</blockquote>`;
      case "figure": return `<figure><img src="${esc(b.src)}" alt="${esc(b.caption || "")}"><figcaption>${esc(b.caption || "")}</figcaption></figure>`;
      case "compare": case "cols": {
        const cols = b.cols || [b.left, b.right].filter(Boolean).map(c => [{type: "heading", text: c.title || ""}, ...(c.blocks || [])]);
        return (b.title ? `<h3>${esc(b.title)}</h3>` : "") + `<div class="cmp">${cols.map(c => `<div class="cmpc">${c.map(x => visual(x, nodeLink)).join("")}</div>`).join("")}</div>`;
      }
      default: return `<p class="muted">Không vẽ được block loại "${esc(b.type)}"</p>`;
    }
  }

  // file:// pages can throw on localStorage in some browsers; progress then lasts only for the session.
  const mem = {};
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return k in mem ? mem[k] : d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { mem[k] = v; } },
  };

  function math(root) {
    if (window.renderMathInElement) {
      renderMathInElement(root || document.body, {
        delimiters: [{left: "$$", right: "$$", display: true}, {left: "$", right: "$", display: false}],
        throwOnError: false,
      });
    }
  }

  async function copy(text, btn) {
    try { await navigator.clipboard.writeText(text); }
    catch {
      const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta); ta.select();
      document.execCommand("copy"); ta.remove();
    }
    if (btn) { const old = btn.textContent; btn.textContent = "Đã copy"; setTimeout(() => btn.textContent = old, 1500); }
  }

  return {esc, md, flow, bars, table, formula, visual, store, math, copy};
})();
