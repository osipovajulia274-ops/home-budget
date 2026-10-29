/* Домашняя бухгалтерия — клиентская часть. Данные приходят из /api/*.
 * Все данные пользователя (названия позиций, магазины) попадают в страницу только через textContent,
 * innerHTML не используется. Графики — встроенный SVG (без внешних библиотек). */
"use strict";
(() => {
  const MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"];
  const MONTHS_SHORT = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];
  const CAT_ORDER = ["food", "eating_out", "housing", "transport", "health", "home", "clothes_beauty", "family_kids",
    "pets", "leisure", "education", "subscriptions", "other"];
  const LEVELS = {
    ok: "в норме", ahead_of_pace: "тратите быстрее, чем идёт месяц", watch: "почти исчерпан", exceeded: "превышен",
  };
  const SAMPLE_RECEIPT = `ООО "Продукты Плюс" (пример)
Кассовый чек / ПРИХОД
Дата: 27.10.2026 18:42
--------------------------------
МОЛОКО ПАСТ. 3,2% 900МЛ  2 x 89.90 =179.80
ХЛЕБ БОРОДИНСКИЙ НАРЕЗ.  1 x 54.00 =54.00
СЫР РОССИЙСКИЙ 200Г  1 x 189.99 =189.99
КУРИЦА ФИЛЕ ОХЛ. 1КГ  1 x 349.50 =349.50
ЯЙЦО КУР. С1 10ШТ  2 x 96.80 =193.60
ПЕЧЕНЬЕ ЮБИЛЕЙНОЕ 112Г  3 x 34.90 =104.70
ВОДА ПИТЬЕВАЯ Н/Г 1.5Л  2 x 39.00 =78.00
--------------------------------
ИТОГ 1149.59
НАЛИЧНЫМИ 1500.00`;

  // ------------------------------------------------------------------ helpers ----
  const $ = (sel, root = document) => root.querySelector(sel);

  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (k === "value" || k === "checked" || k === "disabled" || k === "selected") el[k] = v;
      else el.setAttribute(k, v === true ? "" : v);
    }
    append(el, kids);
    return el;
  }
  function append(el, kids) {
    for (const k of kids.flat(Infinity)) {
      if (k === null || k === undefined || k === false) continue;
      el.append(k instanceof Node ? k : document.createTextNode(String(k)));
    }
  }
  const SVGNS = "http://www.w3.org/2000/svg";
  function s(tag, attrs, ...kids) {
    const el = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
    append(el, kids);
    return el;
  }

  const nf0 = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 0 });
  const nf2 = new Intl.NumberFormat("ru-RU", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const rub = (n) => nf0.format(Math.round(n)) + " ₽";
  const rub2 = (n) => nf2.format(n) + " ₽";
  const pct = (x) => Math.round(x * 100) + "%";
  const signedPct = (x) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(Math.round(x * 100)) + "%";
  const pad = (n) => String(n).padStart(2, "0");
  const fmtDate = (iso) => (iso ? `${iso.slice(8, 10)}.${iso.slice(5, 7)}` : "");
  const num = (str) => parseFloat(String(str).replace(/\s/g, "").replace(",", "."));
  const monthLabel = (key) => { const [y, m] = key.split("-").map(Number); return `${MONTHS[m - 1]} ${y}`; };
  function shiftMonth(key, k) {
    let [y, m] = key.split("-").map(Number);
    m += k - 1; y += Math.floor(m / 12); m = ((m % 12) + 12) % 12 + 1;
    return `${y}-${pad(m)}`;
  }
  const curMonthLocal = () => { const d = new Date(); return `${d.getFullYear()}-${pad(d.getMonth() + 1)}`; };
  const monthParam = () => {
    const m = new URLSearchParams(location.search).get("month");
    return /^\d{4}-\d{2}$/.test(m || "") ? m : null;
  };
  const qs = (m) => (m ? `?month=${m}` : "");
  function goMonth(key) { const u = new URL(location.href); u.searchParams.set("month", key); location.href = u.toString(); }
  const isPartialMonth = (sum) => {
    const [y, m] = sum.month.split("-").map(Number);
    return Number(sum.as_of.slice(8, 10)) < new Date(y, m, 0).getDate();
  };
  const catColor = (id) => `var(--c${Math.max(CAT_ORDER.indexOf(id), 0) + 1})`;

  async function api(path, opts = {}) {
    const { json, ...rest } = opts;
    const init = { credentials: "same-origin", ...rest };
    if (json !== undefined) {
      init.method = opts.method || "POST";
      init.headers = { "Content-Type": "application/json" };
      init.body = JSON.stringify(json);
    }
    const r = await fetch(path, init);
    if (r.status === 204) return null;
    let data = null;
    try { data = await r.json(); } catch (e) { /* no body */ }
    if (!r.ok) {
      const d = data && data.detail;
      throw new Error(typeof d === "string" ? d : Array.isArray(d) ? "Проверьте введённые данные" : `Ошибка ${r.status}`);
    }
    return data;
  }
  let _tax = null;
  async function taxonomy() { return (_tax = _tax || (await api("/api/taxonomy"))); }

  const spinner = () => h("div", { class: "spinner", role: "status" }, "Загрузка…");
  const banner = (kind, text) => h("div", { class: `banner ${kind}`, role: kind === "err" ? "alert" : "status" }, text);
  function errorCard(e, retry) {
    return h("div", { class: "card" }, banner("err", e.message || "Что-то пошло не так"),
      retry ? h("button", { onclick: retry }, "Повторить") : null);
  }
  function mount(fn) {
    const root = $("#app");
    const run = async () => {
      root.replaceChildren(spinner());
      try { await fn(root); } catch (e) { root.replaceChildren(errorCard(e, run)); }
    };
    return run();
  }

  function catSelect(tax, value, { auto = false, label = "Категория" } = {}) {
    const sel = h("select", { "aria-label": label });
    if (auto) sel.append(h("option", { value: "", text: "Определить автоматически" }));
    for (const c of tax) {
      const og = h("optgroup", { label: c.title });
      for (const sub of c.subcategories) og.append(h("option", { value: sub.id, text: sub.title }));
      sel.append(og);
    }
    sel.value = value || "";
    return sel;
  }
  function scopeSelect(tax, value) {
    const sel = h("select", { "aria-label": "На что лимит" }, h("option", { value: "total", text: "Все траты (общий лимит)" }));
    const cats = h("optgroup", { label: "Категории" });
    const subs = h("optgroup", { label: "Подкатегории" });
    for (const c of tax) {
      cats.append(h("option", { value: c.id, text: c.title }));
      for (const sub of c.subcategories) subs.append(h("option", { value: sub.id, text: `${c.title} › ${sub.title}` }));
    }
    sel.append(cats, subs);
    sel.value = value || "total";
    return sel;
  }

  function monthBar(key) {
    return h("div", { class: "monthbar" },
      h("button", { "aria-label": "Предыдущий месяц", onclick: () => goMonth(shiftMonth(key, -1)) }, "‹"),
      h("span", { class: "label" }, monthLabel(key)),
      h("button", { "aria-label": "Следующий месяц", disabled: key >= curMonthLocal(), onclick: () => goMonth(shiftMonth(key, 1)) }, "›"));
  }

  // ------------------------------------------------------------------- charts ----
  function donut(parts, top, bottom) {
    const total = parts.reduce((a, p) => a + p.value, 0);
    const svg = s("svg", { viewBox: "0 0 42 42", role: "img", "aria-label": "Структура трат по категориям" });
    svg.append(s("circle", { cx: 21, cy: 21, r: 15.9155, fill: "none", "stroke-width": 5, style: "stroke:var(--surface-2)" }));
    let offset = 25;
    for (const p of parts) {
      if (p.value <= 0) continue;
      const frac = (p.value / total) * 100;
      svg.append(s("circle", {
        cx: 21, cy: 21, r: 15.9155, fill: "none", "stroke-width": 5,
        "stroke-dasharray": `${frac} ${100 - frac}`, "stroke-dashoffset": offset, style: `stroke:${p.color}`,
      }, s("title", {}, `${p.label}: ${rub(p.value)}`)));
      offset -= frac;
    }
    svg.append(s("text", { x: 21, y: 20.2, "text-anchor": "middle", class: "donut-label" }, top),
      s("text", { x: 21, y: 25, "text-anchor": "middle", class: "donut-value" }, bottom));
    return h("div", { class: "donut-wrap" }, svg);
  }
  function bars(items, ariaLabel) {
    const W = 300, H = 112, base = 84, top = 18;
    const max = Math.max(...items.map((i) => i.value), 1);
    const slot = W / items.length, bw = Math.min(slot * 0.6, 34);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart bars", role: "img", "aria-label": ariaLabel });
    items.forEach((it, i) => {
      const x = slot * i + (slot - bw) / 2, hh = Math.max((it.value / max) * (base - top), it.value > 0 ? 2 : 0);
      svg.append(s("rect", { x, y: base - hh, width: bw, height: hh, rx: 3, class: it.cur ? "cur" : "" },
        s("title", {}, `${it.title || it.label}: ${rub(it.value)}`)));
      if (it.value > 0) svg.append(s("text", { x: x + bw / 2, y: base - hh - 4 }, it.value >= 10000 ? `${Math.round(it.value / 1000)}к` : nf0.format(Math.round(it.value))));
      svg.append(s("text", { x: x + bw / 2, y: base + 14 }, it.label));
    });
    return svg;
  }

  // -------------------------------------------------------------- shared cards ----
  function recoLink(r) {
    switch (r.rule_id) {
      case "R03_category_growth": case "R13_seasonal_prep": return r.category ? `/goals?scope=${r.category}` : null;
      case "R10_subscriptions_found": case "R11_unused_gym": return "/subscriptions";
      default: return r.category ? `/categories?open=${r.category}` : "/categories";
    }
  }
  function recoCard(r, onHide) {
    const link = recoLink(r);
    return h("article", { class: `card reco ${r.severity}` },
      h("h3", {}, r.title), h("p", { class: "body" }, r.body),
      h("div", { class: "row" },
        link && r.action ? h("a", { class: "btn small", href: link }, r.action) : null,
        onHide ? h("button", { class: "small ghost", onclick: () => onHide(r) }, "Скрыть") : null));
  }
  async function hideReco(r, card) {
    try { await api(`/api/recommendations/${r.id}/dismiss`, { method: "POST" }); card.remove(); }
    catch (e) { card.append(banner("err", e.message)); }
  }
  function goalCard(g) {
    const used = Math.min(g.used_share, 1);
    return h("li", {},
      h("div", { class: "item-head" }, h("span", { class: "t" }, g.title), h("span", { class: `pill ${g.level}` }, LEVELS[g.level])),
      h("div", { class: `bar ${g.level}`, role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": Math.round(used * 100) }, h("i", { style: `width:${used * 100}%` })),
      h("div", { class: "small muted" },
        `Потрачено ${rub(g.spent)} из ${rub(g.limit_amount)} (${pct(g.used_share)}). `,
        g.over_by > 0 ? `Превышено на ${rub(g.over_by)}.` : `Осталось ${rub(g.remaining)}${g.days_left > 0 ? `, примерно ${rub(g.per_day_left)} в день` : ""}.`));
  }

  // --------------------------------------------------------------------- home ----
  function home() {
    return mount(async (root) => {
      const m = monthParam(), q = qs(m);
      const [sum, hist, goals, recs] = await Promise.all([
        api("/api/summary" + q), api("/api/history" + q), api("/api/goals" + q), api("/api/recommendations" + q)]);
      const out = [monthBar(sum.month)];
      const empty = sum.total === 0 && sum.prev_month_total === null;

      out.push(h("section", { class: "card hero" },
        h("h2", {}, `Потрачено · ${monthLabel(sum.month)}`),
        h("div", { class: "big num" }, rub(sum.total)),
        h("div", { class: "sub" }, empty ? "Пока нет данных за этот месяц" : `по ${fmtDate(sum.as_of)}, дней с тратами: ${sum.days_with_data}`),
        h("div", { class: "actions" },
          h("a", { class: "btn primary", href: "/add" }, "Добавить траты"),
          empty ? h("button", { onclick: async (e) => { e.target.disabled = true; try { await api("/api/demo/load", { method: "POST" }); location.reload(); } catch (err) { e.target.disabled = false; alert(err.message); } } }, "Загрузить демо-данные") : null)));
      if (empty) { root.replaceChildren(...out); return; }

      const cmp = h("section", { class: "card" }, h("h2", {}, "Сравнение с вашими прошлыми месяцами"));
      if (hist.pace) {
        const d = hist.pace.difference;
        cmp.append(
          h("div", { class: "kv" }, h("span", {}, `К ${fmtDate(hist.as_of)} обычно`), h("b", { class: "num" }, rub(hist.pace.usual_by_now))),
          h("div", { class: "kv" }, h("span", {}, "Сейчас"), h("b", { class: "num" }, rub(hist.pace.spent_now))),
          h("div", { class: "kv" }, h("span", {}, "Разница"), h("b", { class: `num ${d > 0 ? "delta-up" : "delta-down"}` }, (d > 0 ? "+" : d < 0 ? "−" : "") + rub(Math.abs(d)))),
          h("p", { class: "small muted" }, `«Обычно» — медиана за ${hist.pace.n_months} ${hist.pace.n_months === 1 ? "полный месяц" : "полных месяца"} к тому же числу. Это сравнение фактов, а не прогноз.`));
      }
      if (hist.typical) {
        cmp.append(h("div", { class: "kv" }, h("span", {}, "Типичный месяц"), h("b", { class: "num" }, rub(hist.typical.median))),
          h("div", { class: "kv" }, h("span", {}, "Размах"), h("span", { class: "num" }, `${rub(hist.typical.lowest)} – ${rub(hist.typical.highest)}`)));
      }
      if (hist.months.length) {
        const key = hist.month;
        cmp.append(bars(hist.months.map((x) => ({ label: MONTHS_SHORT[Number(x.month.slice(5)) - 1], value: x.total, cur: x.month === key, title: monthLabel(x.month) })), "Траты по месяцам"));
      }
      if (hist.note) cmp.append(h("p", { class: "small muted" }, hist.note));
      out.push(cmp);

      const gc = h("section", { class: "card" }, h("h2", {}, "Цели на месяц"));
      if (goals.length) gc.append(h("ul", { class: "list" }, goals.map(goalCard)), h("a", { class: "btn small", href: "/goals" }, "Управлять целями"));
      else gc.append(h("p", { class: "muted" }, "Лимитов пока нет. Задайте лимит на месяц — приложение покажет, сколько осталось."), h("a", { class: "btn small", href: "/goals" }, "Добавить цель"));
      out.push(gc);

      out.push(h("h2", {}, "Главное"));
      if (recs.length) {
        out.push(...recs.slice(0, 3).map((r) => { const c = recoCard(r, null); return c; }),
          h("a", { class: "btn", href: "/recommendations" }, "Все советы"));
      } else out.push(h("p", { class: "muted" }, "Советов пока нет — для них нужно больше данных."));
      root.replaceChildren(...out);
    });
  }

  // ---------------------------------------------------------------------- add ----
  function add() {
    return mount(async (root) => {
      const tax = await taxonomy();
      const tabs = [["text", "Текст чека"], ["csv", "CSV из банка"], ["manual", "Вручную"]];
      const panels = {};
      const seg = h("div", { class: "seg", role: "tablist" });
      const result = h("div", { "aria-live": "polite" });
      const select = (key) => {
        for (const [k] of tabs) { panels[k].classList.toggle("hidden", k !== key); seg.querySelector(`[data-tab="${k}"]`).setAttribute("aria-selected", String(k === key)); }
        result.replaceChildren();
      };
      for (const [k, label] of tabs) seg.append(h("button", { role: "tab", "data-tab": k, onclick: () => select(k) }, label));
      panels.text = textPanel(tax, result); panels.csv = csvPanel(tax, result); panels.manual = manualPanel(tax, result);
      root.replaceChildren(seg, panels.text, panels.csv, panels.manual, result);
      select("text");
    });
  }

  function textPanel(tax, result) {
    const ta = h("textarea", { id: "receipt-text", placeholder: "Вставьте сюда текст кассового чека…", "aria-label": "Текст чека" });
    const date = h("input", { type: "date", id: "receipt-date", "aria-label": "Дата чека, если её нет в тексте" });
    const go = h("button", { class: "primary" }, "Разобрать");
    go.addEventListener("click", async () => {
      if (!ta.value.trim()) { result.replaceChildren(banner("err", "Вставьте текст чека")); return; }
      go.disabled = true; result.replaceChildren(spinner());
      try {
        const p = await api("/api/import/receipt-text", { json: { text: ta.value, date: date.value || null } });
        renderPreview(result, { kind: "text", tax, meta: p, items: p.items.map((i) => ({ ...i, sub: i.subcategory, include: true })) });
      } catch (e) { result.replaceChildren(banner("err", e.message)); } finally { go.disabled = false; }
    });
    return h("section", { class: "card" },
      h("div", { class: "field" }, h("label", { for: "receipt-text" }, "Текст чека"), ta),
      h("div", { class: "field" }, h("label", { for: "receipt-date" }, "Дата (нужна, только если её нет в тексте)"), date),
      h("div", { class: "actions" }, go, h("button", { class: "ghost", onclick: () => { ta.value = SAMPLE_RECEIPT; } }, "Вставить пример")),
      h("p", { class: "small muted" }, "ИНН и имя ИП не сохраняются. Нужны строки вида «Название 2 x 89.90 =179.80»; ИТОГ используется для сверки."));
  }

  function csvPanel(tax, result) {
    const file = h("input", { type: "file", id: "csv-file", accept: ".csv,.txt,text/csv", "aria-label": "Файл CSV" });
    let current = null;
    const send = async (mapping) => {
      result.replaceChildren(spinner());
      const fd = new FormData();
      fd.append("file", current);
      if (mapping) fd.append("mapping", JSON.stringify(mapping));
      try {
        const p = await api("/api/import/csv", { method: "POST", body: fd });
        renderPreview(result, { kind: "csv", tax, meta: p, items: p.items.map((i) => ({ ...i, sub: i.subcategory, include: true })), remap: send });
      } catch (e) { result.replaceChildren(banner("err", e.message)); }
    };
    file.addEventListener("change", () => { current = file.files[0] || null; if (current) send(null); });
    return h("section", { class: "card" },
      h("div", { class: "field" }, h("label", { for: "csv-file" }, "Выписка банка (CSV)"), file),
      h("p", { class: "small muted" }, "Подойдёт файл из интернет-банка в UTF-8 или Windows-1251. Расходы — отрицательные суммы. Названия колонок угадываются автоматически, их можно поправить после загрузки."));
  }

  function manualPanel(tax, result) {
    const d = new Date();
    const date = h("input", { type: "date", id: "m-date", value: `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` });
    const title = h("input", { type: "text", id: "m-title", maxlength: 300, placeholder: "Например: Яндекс Такси" });
    const amount = h("input", { type: "text", id: "m-amount", inputmode: "decimal", placeholder: "0,00" });
    const cat = catSelect(tax, "", { auto: true });
    const form = h("form", { class: "card" },
      h("div", { class: "field" }, h("label", { for: "m-date" }, "Дата"), date),
      h("div", { class: "field" }, h("label", { for: "m-title" }, "Что куплено"), title),
      h("div", { class: "field" }, h("label", { for: "m-amount" }, "Сумма, ₽"), amount),
      h("div", { class: "field" }, h("label", {}, "Категория"), cat),
      h("div", { class: "actions" }, h("button", { class: "primary", type: "submit" }, "Добавить")));
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const a = num(amount.value);
      if (!title.value.trim() || !(a > 0)) { result.replaceChildren(banner("err", "Укажите название и сумму больше нуля")); return; }
      try {
        const t = await api("/api/transactions", { json: { date: date.value, title: title.value.trim(), amount: a, subcategory: cat.value || null } });
        const sel = catSelect(tax, t.subcategory);
        sel.addEventListener("change", async () => {
          try { await api(`/api/transactions/${t.id}`, { method: "PATCH", json: { subcategory: sel.value } }); note.textContent = "Категория обновлена и будет запомнена."; }
          catch (e) { note.textContent = e.message; }
        });
        const note = h("p", { class: "small muted", role: "status" }, t.category_source === "model" && t.confidence < 0.7 ? "Модель не уверена — проверьте категорию." : "");
        result.replaceChildren(h("div", { class: "card" }, banner("ok", `Добавлено: ${t.title} — ${rub2(t.amount)}`), sel, note,
          h("div", { class: "actions" }, h("a", { class: "btn", href: "/" }, "На главную"))));
        title.value = ""; amount.value = "";
      } catch (e) { result.replaceChildren(banner("err", e.message)); }
    });
    return form;
  }

  function renderPreview(root, st) {
    const m = st.meta, csv = st.kind === "csv";
    const included = () => st.items.filter((i) => i.include);
    const spendSum = () => included().filter((i) => i.sub !== "other.transfers").reduce((a, i) => a + i.amount, 0);
    const listBox = h("div"), footer = h("div");

    const draw = () => {
      listBox.replaceChildren();
      st.items.forEach((it, idx) => {
        if (!it.include) return;
        const review = it.needs_review && it.sub === it.subcategory;
        const sel = catSelect(st.tax, it.sub, { label: `Категория: ${it.title}` });
        sel.addEventListener("change", () => { it.sub = sel.value; draw(); });
        listBox.append(h("div", { class: `prev-item${review ? " review" : ""}` },
          h("div", { class: "item-head" }, h("span", { class: "t" }, it.title), h("b", { class: "num" }, rub2(it.amount))),
          csv ? h("div", { class: "small muted" }, [fmtDate(it.date), it.bank_category ? ` · ${it.bank_category}` : "", it.category_source === "bank" ? " · категория по данным банка" : ""]) : (it.qty && it.qty !== 1 ? h("div", { class: "small muted" }, `Количество: ${it.qty}`) : null),
          review ? h("div", { class: "small", style: "color:var(--warn)" }, `Проверьте категорию (уверенность ${pct(it.confidence)})`) : null,
          h("div", { class: "line2" }, sel, h("button", { class: "small ghost", "aria-label": `Убрать «${it.title}»`, onclick: () => { it.include = false; draw(); } }, "✕"))));
      });
      if (!included().length) listBox.append(h("div", { class: "empty" }, "Все позиции убраны"));
      footer.replaceChildren(h("div", { class: "kv" }, h("span", {}, `Позиций: ${included().length}`), h("b", { class: "num" }, `Итого ${rub2(spendSum())}`)),
        h("div", { class: "actions" },
          h("button", { class: "primary", disabled: !included().length, onclick: save }, "Сохранить"),
          h("button", { class: "ghost", onclick: () => root.replaceChildren() }, "Отмена")));
    };

    async function save(ev) {
      ev.target.disabled = true;
      const removed = st.items.length !== included().length;
      const body = {
        source: st.kind, store: csv ? null : m.store, purchased_at: csv ? null : m.purchased_at, total: csv ? null : m.total,
        sum_matches_total: csv || removed ? null : m.sum_matches_total,
        items: included().map((i) => ({
          title: i.title, amount: i.amount, qty: i.qty ?? null, date: i.date || null, subcategory: i.sub,
          predicted_subcategory: i.subcategory, confidence: i.confidence, category_source: i.category_source })),
      };
      try {
        const r = await api("/api/import/confirm", { json: body });
        root.replaceChildren(h("div", { class: "card" }, banner("ok", `Сохранено позиций: ${r.saved}`),
          h("p", { class: "small muted" }, "Исправленные категории запомнены: такие же названия в следующих чеках определятся по вашему выбору."),
          h("div", { class: "actions" }, h("a", { class: "btn primary", href: "/" }, "На главную"), h("a", { class: "btn", href: "/categories" }, "Категории"), h("a", { class: "btn ghost", href: "/add" }, "Добавить ещё"))));
      } catch (e) { ev.target.disabled = false; footer.append(banner("err", e.message)); }
    }

    const head = h("div", { class: "card" });
    if (csv) {
      head.append(h("h2", {}, `Выписка: ${m.items.length} операций`),
        banner("info", `Расходы без переводов: ${rub2(m.items_sum)}. Поступления (${m.income_count}) на ${rub2(m.income_total)} в траты не входят; переводы между своими счетами тоже.`),
        mappingEditor(st));
    } else {
      head.append(h("h2", {}, `Чек${m.store ? ": " + m.store : ""}`),
        h("p", { class: "small muted" }, m.purchased_at ? `${m.purchased_at.slice(8, 10)}.${m.purchased_at.slice(5, 7)}.${m.purchased_at.slice(0, 4)}` : "Дата не найдена"),
        m.sum_matches_total === true ? banner("ok", `✓ Сумма позиций совпала с ИТОГ (${rub2(m.total)})`)
          : m.sum_matches_total === false ? banner("warn", `Сумма позиций ${rub2(m.items_sum)} не совпала с ИТОГ ${rub2(m.total)}. Проверьте список.`)
            : banner("info", "ИТОГ в тексте не найден — сверить сумму не с чем"));
    }
    for (const w of m.warnings || []) if (!w.includes("не совпала")) head.append(banner("warn", w));
    const nReview = st.items.filter((i) => i.needs_review).length;
    if (nReview) head.append(banner("info", `Проверьте категории у ${nReview} ${nReview === 1 ? "позиции" : "позиций"} — они выделены.`));
    root.replaceChildren(head, h("div", { class: "card" }, listBox, footer));
    if (!st.items.length) { listBox.append(h("div", { class: "empty" }, "Позиции не найдены")); }
    draw();
  }

  function mappingEditor(st) {
    const m = st.meta, cols = m.columns;
    const mk = (key, label) => {
      const sel = h("select", { "aria-label": label }, h("option", { value: "", text: "— нет —" }), cols.map((c) => h("option", { value: c, text: c })));
      sel.value = typeof m.mapping[key] === "string" ? m.mapping[key] : "";
      return h("div", { class: "field" }, h("label", {}, label), sel);
    };
    const date = mk("date", "Дата"), amount = mk("amount", "Сумма"), category = mk("category", "Категория банка"), mcc = mk("mcc", "MCC");
    const descBoxes = cols.map((c) => h("label", {}, h("input", { type: "checkbox", value: c, checked: (m.mapping.description || []).includes(c) }), c));
    return h("details", {}, h("summary", {}, "Колонки файла (если распознаны неверно)"),
      date, amount, h("div", { class: "field" }, h("label", {}, "Описание (можно несколько)"), h("div", { class: "checks" }, descBoxes)), category, mcc,
      h("button", { onclick: () => st.remap({
        date: $("select", date).value || null, amount: $("select", amount).value || null,
        category: $("select", category).value || null, mcc: $("select", mcc).value || null,
        description: descBoxes.map((b) => $("input", b)).filter((i) => i.checked).map((i) => i.value) }) }, "Применить"));
  }

  // --------------------------------------------------------------- categories ----
  function categories() {
    return mount(async (root) => {
      const m = monthParam();
      const [sum, tax] = await Promise.all([api("/api/summary" + qs(m)), taxonomy()]);
      const partial = isPartialMonth(sum);
      const detail = h("div");
      const out = [monthBar(sum.month)];
      if (!sum.by_category.length) {
        out.push(h("div", { class: "card empty" }, "За этот месяц трат нет.", h("div", { class: "actions", style: "justify-content:center" }, h("a", { class: "btn primary", href: "/add" }, "Добавить траты"))));
        root.replaceChildren(...out); return;
      }
      const parts = sum.by_category.map((c) => ({ label: c.title, value: c.amount, color: catColor(c.category) }));
      const list = h("div");
      for (const c of sum.by_category) {
        const sub = partial ? (c.prev_amount !== null ? `в прошлом месяце: ${rub(c.prev_amount)}` : "") : (c.delta_vs_prev !== null ? `к прошлому месяцу: ${signedPct(c.delta_vs_prev)}` : "");
        list.append(h("button", { class: "catrow", "data-cat": c.category, onclick: () => openCat(c.category) },
          h("div", { class: "item-head" }, h("span", { class: "t" }, h("span", { class: "dot", style: `background:${catColor(c.category)}` }), c.title), h("b", { class: "num" }, rub(c.amount))),
          h("div", { class: "small muted" }, `${pct(c.share)} трат${sub ? " · " + sub : ""}`)));
      }
      out.push(h("section", { class: "card" }, h("h2", {}, "Куда уходят деньги"), donut(parts, "Всего", rub(sum.total)), list), detail);
      root.replaceChildren(...out);

      async function openCat(cid) {
        const c = sum.by_category.find((x) => x.category === cid);
        detail.replaceChildren(h("div", { class: "card" }, spinner()));
        const keys = [-5, -4, -3, -2, -1, 0].map((k) => shiftMonth(sum.month, k));
        try {
          const [series, txs] = await Promise.all([
            Promise.all(keys.map((k) => (k === sum.month ? Promise.resolve(sum) : api(`/api/summary?month=${k}`)))),
            api(`/api/transactions?month=${sum.month}&category=${cid}`)]);
          const card = h("section", { class: "card" }, h("h2", {}, c.title),
            h("ul", { class: "list" }, c.subcategories.map((x) => h("li", {}, h("div", { class: "item-head" }, h("span", { class: "t" }, x.title), h("b", { class: "num" }, rub(x.amount)))))),
            h("h3", {}, "По месяцам"),
            bars(series.map((sm, i) => ({ label: MONTHS_SHORT[Number(keys[i].slice(5)) - 1], value: (sm.by_category.find((x) => x.category === cid) || { amount: 0 }).amount, cur: keys[i] === sum.month, title: monthLabel(keys[i]) })), `Траты на «${c.title}» по месяцам`),
            h("h3", {}, "Покупки"));
          const status = h("p", { class: "small muted", role: "status" });
          const ul = h("ul", { class: "list" });
          for (const t of txs) {
            const sel = catSelect(tax, t.subcategory, { label: `Категория: ${t.title}` });
            sel.addEventListener("change", async () => {
              try { await api(`/api/transactions/${t.id}`, { method: "PATCH", json: { subcategory: sel.value } }); status.textContent = "Категория изменена. Итоги обновятся после перезагрузки страницы."; }
              catch (e) { status.textContent = e.message; }
            });
            ul.append(h("li", {}, h("div", { class: "item-head" }, h("span", { class: "t" }, `${fmtDate(t.date)} · ${t.title}`), h("b", { class: "num" }, rub2(t.amount))), sel));
          }
          card.append(ul.children.length ? ul : h("p", { class: "muted" }, "Покупок нет"), status);
          detail.replaceChildren(card);
          card.scrollIntoView({ behavior: "smooth", block: "start" });
        } catch (e) { detail.replaceChildren(errorCard(e, () => openCat(cid))); }
      }
      const open = new URLSearchParams(location.search).get("open");
      if (open && sum.by_category.some((x) => x.category === open)) openCat(open);
    });
  }

  // ---------------------------------------------------------- recommendations ----
  function recommendations() {
    return mount(async (root) => {
      const m = monthParam();
      const recs = await api("/api/recommendations" + qs(m));
      const out = [monthBar(m || curMonthLocal())];
      if (!recs.length) out.push(h("div", { class: "card empty" }, "Советов пока нет. Они появятся, когда накопится больше данных."));
      for (const r of recs) { const holder = h("div"); const c = recoCard(r, (rr) => hideReco(rr, holder)); holder.append(c); out.push(holder); }
      out.push(h("p", { class: "small muted" }, "Советы пересчитываются при каждом открытии. Скрытая карточка не вернётся в этом месяце. Это описание ваших трат и бытовые подсказки, не финансовая консультация."));
      root.replaceChildren(...out);
    });
  }

  // -------------------------------------------------------------------- goals ----
  function goals() {
    return mount(async (root) => {
      const m = monthParam();
      const [list, tax] = await Promise.all([api("/api/goals" + qs(m)), taxonomy()]);
      const prefill = new URLSearchParams(location.search).get("scope");
      const scope = scopeSelect(tax, prefill), limit = h("input", { type: "text", inputmode: "decimal", id: "goal-limit", placeholder: "Например, 30 000" });
      const msg = h("div", { "aria-live": "polite" });
      const form = h("form", { class: "card" }, h("h2", {}, "Новая цель"),
        h("p", { class: "small muted" }, "Лимит на месяц: приложение покажет, сколько потрачено, сколько осталось и в каком темпе идёт месяц. Это не прогноз."),
        h("div", { class: "field" }, h("label", {}, "На что"), scope),
        h("div", { class: "field" }, h("label", { for: "goal-limit" }, "Лимит на месяц, ₽"), limit),
        h("div", { class: "actions" }, h("button", { class: "primary", type: "submit" }, "Добавить цель")), msg);
      form.addEventListener("submit", async (ev) => {
        ev.preventDefault();
        const v = num(limit.value);
        if (!(v > 0)) { msg.replaceChildren(banner("err", "Укажите лимит больше нуля")); return; }
        try { await api("/api/goals", { json: { scope: scope.value, limit_amount: v } }); location.href = "/goals" + qs(m); }
        catch (e) { msg.replaceChildren(banner("err", e.message)); }
      });
      const items = h("ul", { class: "list" });
      for (const g of list) {
        const inp = h("input", { type: "text", inputmode: "decimal", value: String(g.limit_amount).replace(".", ","), "aria-label": `Лимит: ${g.title}` });
        const note = h("span", { class: "small muted", role: "status" });
        const li = goalCard(g);
        li.append(h("div", { class: "row", style: "margin-top:.5rem" }, h("div", { class: "grow" }, inp),
          h("button", { class: "small", onclick: async () => { const v = num(inp.value); if (!(v > 0)) { note.textContent = "Лимит должен быть больше нуля"; return; }
            try { await api(`/api/goals/${g.id}`, { method: "PUT", json: { scope: g.scope, limit_amount: v } }); location.reload(); } catch (e) { note.textContent = e.message; } } }, "Сохранить"),
          h("button", { class: "small danger", onclick: async () => { if (!confirm(`Удалить цель «${g.title}»?`)) return; try { await api(`/api/goals/${g.id}`, { method: "DELETE" }); location.reload(); } catch (e) { note.textContent = e.message; } } }, "Удалить")), note);
        items.append(li);
      }
      root.replaceChildren(monthBar(m || curMonthLocal()),
        h("section", { class: "card" }, h("h2", {}, "Ваши цели"), list.length ? items : h("p", { class: "muted" }, "Целей пока нет.")), form);
    });
  }

  // ------------------------------------------------------------ subscriptions ----
  function subscriptions() {
    return mount(async (root) => {
      const d = await api("/api/subscriptions");
      const subs = d.items.filter((i) => i.category === "subscriptions"), other = d.items.filter((i) => i.category !== "subscriptions");
      const row = (i) => h("li", {}, h("div", { class: "item-head" }, h("span", { class: "t" }, i.title), h("b", { class: "num" }, rub(i.amount))),
        h("div", { class: "small muted" }, `раз в ~${i.interval_days} дн. · списаний: ${i.occurrences} · последнее ${fmtDate(i.last_date)}`));
      const out = [h("section", { class: "card hero" }, h("h2", {}, "Подписки и сервисы"),
        h("div", { class: "big num" }, subs.length ? `${rub(d.subscriptions_per_month)} в месяц` : "—"),
        h("div", { class: "sub" }, subs.length ? `${rub(d.subscriptions_per_year)} в год (если платежи не изменятся)` : "Повторяющихся подписок не найдено"))];
      if (subs.length) out.push(h("section", { class: "card" }, h("ul", { class: "list" }, subs.map(row))));
      if (other.length) out.push(h("section", { class: "card" }, h("h2", {}, "Другие регулярные платежи"), h("ul", { class: "list" }, other.map(row))));
      out.push(h("p", { class: "small muted" }, "Найдено по повторяющимся списаниям: одинаковое название, сумма с отклонением до 5% и интервал 28–31 день. Проверьте, все ли подписки вам нужны."));
      root.replaceChildren(...out);
    });
  }

  // ----------------------------------------------------------------- settings ----
  function settings() {
    return mount(async (root) => {
      const [me, health] = await Promise.all([api("/api/me"), api("/health").catch(() => null)]);
      const income = h("input", { type: "text", inputmode: "decimal", id: "income", value: me.monthly_income ? String(me.monthly_income).replace(".", ",") : "", placeholder: "Не указан" });
      const note = h("div", { "aria-live": "polite" });
      let theme = "auto"; try { theme = localStorage.getItem("theme") || "auto"; } catch (e) { /* private mode */ }
      const themeSel = h("select", { id: "theme", "aria-label": "Тема оформления" }, [["auto", "Как в системе"], ["light", "Светлая"], ["dark", "Тёмная"]].map(([v, t]) => h("option", { value: v, text: t })));
      themeSel.value = theme;
      themeSel.addEventListener("change", () => {
        try { if (themeSel.value === "auto") localStorage.removeItem("theme"); else localStorage.setItem("theme", themeSel.value); } catch (e) { /* ignore */ }
        if (themeSel.value === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = themeSel.value;
      });
      const busy = async (btn, fn) => { btn.disabled = true; try { await fn(); } catch (e) { note.replaceChildren(banner("err", e.message)); btn.disabled = false; } };
      const demoBtn = h("button", { onclick: () => busy(demoBtn, async () => { await api("/api/demo/load", { method: "POST" }); note.replaceChildren(banner("ok", "Демо-данные загружены (синтетика: вымышленная пара, 3 месяца + текущий).")); demoBtn.disabled = false; }) }, "Загрузить демо-данные");
      const delBtn = h("button", { class: "danger", onclick: () => busy(delBtn, async () => {
        if (!confirm("Удалить все ваши траты, цели и исправления? Это нельзя отменить.")) { delBtn.disabled = false; return; }
        await api("/api/me/delete", { method: "POST" }); location.href = "/"; }) }, "Удалить мои данные");
      root.replaceChildren(
        h("section", { class: "card" }, h("h2", {}, "Доход"),
          h("div", { class: "field" }, h("label", { for: "income" }, "Месячный доход, ₽ (по желанию, для справочных сравнений)"), income),
          h("button", { class: "primary", onclick: async () => { const v = income.value.trim() ? num(income.value) : null; if (v !== null && !(v >= 0)) { note.replaceChildren(banner("err", "Введите число")); return; }
            try { await api("/api/me", { method: "PUT", json: { monthly_income: v } }); note.replaceChildren(banner("ok", "Сохранено")); } catch (e) { note.replaceChildren(banner("err", e.message)); } } }, "Сохранить")),
        h("section", { class: "card" }, h("h2", {}, "Оформление"), themeSel),
        h("section", { class: "card" }, h("h2", {}, "Данные"),
          h("p", { class: "small muted" }, "Демо-режим: данные привязаны к этому браузеру (cookie), входа и паролей нет. Другой браузер или очистка cookie — это другой пользователь."),
          h("div", { class: "actions" }, demoBtn, delBtn), note),
        h("section", { class: "card" }, h("h2", {}, "О проекте"),
          h("p", { class: "small" }, `Категории определяет классификатор, обученный с нуля (${health && health.models ? "TF-IDF по символьным n-граммам + логистическая регрессия" : "модель недоступна"}). Предобученные модели не используются. Неуверенные позиции помечаются для проверки.`),
          h("ul", { class: "small list" },
            h("li", {}, "Open Food Facts, Open Products Facts, Open Beauty Facts, Open Pet Food Facts — названия товаров для обучения. Лицензии: ODbL (база), DbCL (содержимое)."),
            h("li", {}, "Receipt OCR RU (CDEK), лицензия MIT, — реальные чеки, использованы только для проверки качества."),
            h("li", {}, "Росстат, обследование бюджетов домашних хозяйств, — справочные доли расходов."),
            h("li", {}, "Демо-данные и часть обучающих примеров — синтетические.")),
          h("p", { class: "small muted" }, "Приложение описывает ваши траты и даёт бытовые подсказки. Это не финансовая и не инвестиционная консультация.")));
    });
  }

  const pages = { home, add, categories, recommendations, goals, subscriptions, settings };
  const run = pages[document.body.dataset.page];
  if (run) run();
})();
