import html
import os
import re
import sys
import threading
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, "src")
load_dotenv()
from live import load_cached, list_cached, run_brand

MAX_PER_SESSION = 2    # new (uncached) analyses per visitor session
MAX_PER_HOUR = 6       # new analyses across all visitors
MAX_PER_DAY = 25
LEAN_BELOW = 3         # a strength/weakness winning by less than this many times shows as "leaning"
EXAMPLES = ["Sony WH-1000XM5", "Tesla Model 3", "iPhone 17", "Nike shoes"]
ICON = {"strength": "🟢", "lean-pos": "🟢", "weakness": "🔴", "lean-neg": "🟠", "mixed": "🟡"}

st.set_page_config(page_title="What do people say?", page_icon="💬", layout="wide")

CSS = """
<style>
.block-container{max-width:1100px;padding-top:2rem}
#MainMenu, footer{visibility:hidden}
.hero{padding:1.6rem 0 .4rem}
.kicker{font-size:.8rem;letter-spacing:.14em;text-transform:uppercase;color:#a78bfa;font-weight:650}
.hero h1{font-size:2.7rem;line-height:1.15;margin:.35rem 0 .7rem;font-weight:800;
  background:linear-gradient(90deg,#c4b5fd,#7c5cff 45%,#38bdf8);-webkit-background-clip:text;
  background-clip:text;color:transparent}
.sub{opacity:.75;font-size:1.05rem;max-width:740px;line-height:1.5}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:.8rem 0 .9rem}
.kpi{border:1px solid rgba(128,128,128,.25);background:linear-gradient(160deg,rgba(124,92,255,.14),rgba(56,189,248,.05));
  border-radius:14px;padding:14px 18px}
.kpi .v{font-size:1.7rem;font-weight:750;text-transform:capitalize}
.kpi .l{font-size:.82rem;opacity:.7;margin-top:2px}
.vids{font-size:.88rem;opacity:.85;margin:0 0 1rem;line-height:1.5}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px;margin:.6rem 0 1rem}
.card{border:1px solid rgba(128,128,128,.25);background:rgba(128,128,128,.07);border-radius:14px;padding:14px 16px}
.card .top{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;gap:8px}
.card .name{font-weight:650;text-transform:capitalize}
.badge{font-size:.68rem;padding:2px 9px;border-radius:99px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}
.badge.strength{background:rgba(34,197,94,.18);color:#4ade80}
.badge.weakness{background:rgba(239,68,68,.18);color:#f87171}
.badge.mixed{background:rgba(245,158,11,.18);color:#fbbf24}
.badge.lean-pos{background:rgba(163,230,53,.16);color:#a3e635}
.badge.lean-neg{background:rgba(251,146,60,.18);color:#fb923c}
.bar{display:flex;height:9px;border-radius:99px;overflow:hidden;background:rgba(128,128,128,.2)}
.bar i{display:block;height:100%}
.bar .p{background:#22c55e}.bar .u{background:#94a3b8}.bar .n{background:#ef4444}
.meta{margin-top:10px;font-size:.88rem}
.legend{display:flex;gap:14px;font-size:.78rem;opacity:.8;margin-top:4px}
.legend span::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
.legend .lp::before{background:#22c55e}.legend .ln::before{background:#ef4444}
.tag{font-size:.68rem;padding:1px 7px;border-radius:6px;border:1px solid rgba(128,128,128,.45);margin-left:6px;opacity:.85}
.cm{border-left:3px solid #94a3b8;background:rgba(128,128,128,.07);border-radius:0 10px 10px 0;
  padding:9px 14px;margin:8px 0;font-size:.92rem;line-height:1.45}
.cm.positive{border-left-color:#22c55e}.cm.negative{border-left-color:#ef4444}
.cmh{display:flex;gap:10px;font-size:.72rem;opacity:.7;margin-bottom:3px;text-transform:uppercase;letter-spacing:.04em}
.step{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:16px 18px;background:rgba(128,128,128,.05);height:100%}
.step .n{font-size:.75rem;color:#a78bfa;font-weight:700;letter-spacing:.1em}
.step .t{font-weight:650;margin:4px 0 4px;font-size:1.02rem}
.step .d{opacity:.72;font-size:.9rem;line-height:1.45}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def shared():
    return {"lock": threading.Lock(), "runs": []}


S = shared()


# ---------- helpers ----------
def esc_md(t):
    return (t or "").replace("$", "\\$")   # stop "$15,000 ... $25,000" turning into math


def split_sections(text):
    out, cur, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^##\s+(.*)", line)
        if m:
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1).strip(), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


def get_section(sections, prefix):
    for k, v in sections.items():
        if k.lower().startswith(prefix.lower()):
            return v
    return ""


def parse_aspects(sections):
    """aspect -> {'complaints': text, 'praise': text}, read from the Strengths/Weaknesses/Mixed sections."""
    data = {}
    for key in ("Strengths", "Weaknesses", "Mixed"):
        cur = None
        for line in get_section(sections, key).splitlines():
            m = re.match(r"^- \*\*(.+?)\*\*", line)
            if m:
                cur = m.group(1).strip().lower()
                data.setdefault(cur, {"complaints": "", "praise": ""})
                continue
            s = line.strip()
            if cur and s.startswith("- Complaints:"):
                data[cur]["complaints"] = s[len("- Complaints:"):].strip()
            elif cur and s.startswith("- Praise:"):
                data[cur]["praise"] = s[len("- Praise:"):].strip()
    return data


def overview_counts(overview):
    m = re.search(r"covers (\d+) YouTube comments.*?(\d+) were judged relevant.*?(\d+) were assigned",
                  overview, re.S)
    return tuple(int(x) for x in m.groups()) if m else None


def verdict_view(r):
    """Display-only: soften a strength/weakness that wins by less than LEAN_BELOW times."""
    v, pos, neg = r["verdict"], int(r["pos_n"]), int(r["neg_n"])
    if v == "strength" and neg > 0 and pos < LEAN_BELOW * neg:
        return "lean-pos", "leaning positive"
    if v == "weakness" and pos > 0 and neg < LEAN_BELOW * pos:
        return "lean-neg", "leaning negative"
    return v, v


def bar_only(r):
    pos, neg = int(r["pos_pct"]), int(r["neg_pct"])
    neu = max(0, 100 - pos - neg)
    return (f'<div class="bar"><i class="p" style="width:{pos}%"></i><i class="u" style="width:{neu}%"></i>'
            f'<i class="n" style="width:{neg}%"></i></div>'
            f'<div class="legend" style="margin:6px 0 16px"><span class="lp">{int(r["pos_n"])} positive</span>'
            f'<span class="ln">{int(r["neg_n"])} negative</span></div>')


def kpi(label, value):
    return f'<div class="kpi"><div class="v">{html.escape(str(value))}</div><div class="l">{label}</div></div>'


def card(r):
    pos, neg = int(r["pos_pct"]), int(r["neg_pct"])
    neu = max(0, 100 - pos - neg)
    cls, label = verdict_view(r)
    thin = '<span class="tag">thin evidence</span>' if r["evidence"] == "thin" else ""
    return (f'<div class="card"><div class="top"><span class="name">{html.escape(str(r["aspect"]))}</span>'
            f'<span class="badge {cls}">{label}</span></div>'
            f'<div class="bar"><i class="p" style="width:{pos}%"></i><i class="u" style="width:{neu}%"></i>'
            f'<i class="n" style="width:{neg}%"></i></div>'
            f'<div class="meta"><b>{int(r["mentions"])}</b> mentions{thin}</div>'
            f'<div class="legend"><span class="lp">{int(r["pos_n"])} positive</span>'
            f'<span class="ln">{int(r["neg_n"])} negative</span></div></div>')


def cited_rows(body, flt, asp_flt="All"):
    rows = []
    for line in body.splitlines():
        m = re.match(r"^- (C\d+) \(([^,]+), (.+?)\): (.*)$", line)
        if not m:
            continue
        cid, sent, asp, txt = m.groups()
        if asp_flt != "All" and asp.lower() != asp_flt.lower():
            continue
        if flt != "All" and sent != flt.lower():
            continue
        rows.append(f'<div class="cm {sent}"><div class="cmh"><span>{cid}</span>'
                    f'<span>{html.escape(asp)}</span><span>{sent}</span></div>{html.escape(txt)}</div>')
    return "".join(rows)


def videos_block(res):
    vids = res.get("videos") or []
    if not vids:
        st.caption("Video list not available for this older run. Run the brand again to see it.")
        return
    names = " · ".join(html.escape(v["title"][:60] + ("…" if len(v["title"]) > 60 else "")) for v in vids[:3])
    more = " …" if len(vids) > 3 else ""
    st.markdown(f'<div class="vids"><b>Based on comments from {len(vids)} videos:</b> {names}{more}</div>',
                unsafe_allow_html=True)
    with st.expander(f"All {len(vids)} videos analyzed"):
        st.dataframe(pd.DataFrame(vids).rename(columns={"title": "Video", "comments": "Comments",
                                                         "relevant": "About the product"}),
                     hide_index=True)


def pick(name):
    st.session_state["brand_input"] = name
    st.session_state["auto"] = True


def analyze(brand, strict=False):
    brand = " ".join(brand.split())
    if not re.fullmatch(r"[\w .+&'-]{2,40}", brand):
        st.error("Enter 2 to 40 characters: letters, numbers, spaces, and . + & ' - only.")
        return None
    res = load_cached(brand, strict)
    if res is not None:
        return res
    now = time.time()
    S["runs"] = [t for t in S["runs"] if now - t < 86400]
    last_hour = sum(1 for t in S["runs"] if now - t < 3600)
    used = st.session_state.get("used", 0)
    if not (os.getenv("GEMINI_API_KEY") and os.getenv("YOUTUBE_API_KEY")):
        st.error("The server is missing GEMINI_API_KEY or YOUTUBE_API_KEY.")
        return None
    if used >= MAX_PER_SESSION:
        st.warning(f"Limit reached: {MAX_PER_SESSION} new analyses per session. Instant brands still work.")
        return None
    if last_hour >= MAX_PER_HOUR or len(S["runs"]) >= MAX_PER_DAY:
        st.warning("The demo hit its hourly or daily limit to protect API quota. "
                   "Try again later, or open an instant brand.")
        return None
    if not S["lock"].acquire(blocking=False):
        st.warning("Another analysis is running. Try again in a few minutes.")
        return None
    try:
        S["runs"].append(now)
        st.session_state["used"] = used + 1
        with st.status(f"Analyzing '{brand}'. A new brand takes a few minutes.", expanded=True) as status:
            res = run_brand(brand, progress=status.write, strict=strict)
            status.update(label="Done", state="complete", expanded=False)
        return res
    except Exception as e:
        st.error(f"Analysis failed: {type(e).__name__}: {e}")
        return None
    finally:
        S["lock"].release()


def render(res):
    sections = split_sections(res["text"])
    stats = pd.DataFrame(res["stats"])
    st.markdown(f"## {html.escape(res['brand'])}")
    st.caption(f"Generated {res['created']}")
    if res.get("strict"):
        st.caption("Strict mode: videos comparing several products were left out.")

    if not sections or stats.empty:
        st.markdown(esc_md(res["text"]))
        return

    counts = overview_counts(get_section(sections, "Overview"))
    top = stats.loc[stats["mentions"].idxmax(), "aspect"]
    k = []
    if counts:
        k += [kpi("comments analyzed", counts[0]), kpi("about the product", counts[1])]
    k += [kpi("aspects found", len(stats)), kpi("most discussed", top)]
    st.markdown('<div class="kpis">' + "".join(k) + "</div>", unsafe_allow_html=True)
    videos_block(res)

    t1, t2, t3, t4 = st.tabs(["At a glance", "What people say", "Suggestions", "Evidence"])

    with t1:
        st.markdown('<div class="grid">' + "".join(card(r) for _, r in stats.iterrows()) + "</div>",
                    unsafe_allow_html=True)
        st.caption("Bars show the share of positive (green), neutral (grey) and negative (red) comments per aspect. "
                   f"'Leaning' means the verdict rule is met but by less than {LEAN_BELOW}x, so read it as a tendency.")
        st.info("A search for a brand can cover several models (for example a base model and a Pro). "
                "Check the video list above: if the titles are not about the product you meant, "
                "treat the results with caution.")
        with st.expander("How to read these numbers"):
            st.markdown(esc_md(get_section(sections, "Overview")))
        with st.expander("Raw table"):
            st.dataframe(stats, hide_index=True)

    with t2:
        st.caption("Ordered by number of mentions. Ids like C12 match the Evidence tab.")
        notes_by = parse_aspects(sections)
        for i, (_, r) in enumerate(stats.sort_values("mentions", ascending=False).iterrows()):
            cls, label = verdict_view(r)
            d = notes_by.get(str(r["aspect"]).lower(), {})
            title = f"{ICON[cls]} {str(r['aspect']).title()}  ·  {int(r['mentions'])} mentions  ·  {label}"
            with st.expander(title, expanded=(i == 0)):
                st.markdown(bar_only(r), unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.markdown("**What people criticize**")
                c1.markdown(esc_md(d.get("complaints")) or "_Nothing cited._")
                c2.markdown("**What people like**")
                c2.markdown(esc_md(d.get("praise")) or "_Nothing cited._")

    with t3:
        st.info("These are hypotheses drawn from a small set of quoted comments, not validated advice.")
        st.markdown(esc_md(get_section(sections, "Suggestions")) or "None")
        st.caption("Aspects rated as strengths, and aspects with fewer than 3 negative comments, get no suggestion.")

    with t4:
        flt = st.radio("Show", ["All", "Negative", "Positive"], horizontal=True, label_visibility="collapsed")
        asp_opts = ["All"] + [str(a).title() for a in stats.sort_values("mentions", ascending=False)["aspect"]]
        asp_flt = st.selectbox("Filter by aspect", asp_opts)
        rows = cited_rows(get_section(sections, "Cited comments"), flt, asp_flt)
        st.markdown(rows or "No comments match this filter.", unsafe_allow_html=True)

    if res["notes"]:
        with st.expander("Automatic checks that flagged something"):
            for n in res["notes"]:
                st.write("- " + n)
    st.caption("Some off-topic or other-model comments can still pass the filters.")


# ---------- page ----------
st.markdown(
    '<div class="hero"><div class="kicker">YouTube comment analysis</div>'
    '<h1>What do people really say about a product?</h1>'
    '<div class="sub">Type any brand or product. We read YouTube comments, keep the ones about the product itself, '
    'score sentiment per aspect, and write a report where every claim points to a real comment.</div></div>',
    unsafe_allow_html=True)

with st.form("search", border=False):
    c1, c2 = st.columns([5, 1])
    brand_in = c1.text_input("Brand or product", key="brand_input", label_visibility="collapsed",
                             placeholder="e.g. Sony WH-1000XM5, Mahindra Thar, Nike shoes")
    submitted = c2.form_submit_button("Analyze", type="primary")

st.checkbox("Include comparison videos", value=True, key="include_cmp",
            help="On: more comments, but videos comparing several models can mix other products in. "
                 "Off: only videos mainly about the exact product you typed.")

cached = list_cached()
if cached:
    st.caption("Instant results (already analyzed)")
    cols = st.columns(min(len(cached), 6))
    for i, name in enumerate(cached[:6]):
        cols[i].button(name, key=f"c_{i}", on_click=pick, args=(name,))
fresh = [e for e in EXAMPLES if e.lower() not in {c.lower() for c in cached}][:4]
if fresh:
    st.caption("Or try something new (takes a few minutes)")
    cols = st.columns(len(fresh))
    for i, name in enumerate(fresh):
        cols[i].button(name, key=f"e_{i}", on_click=pick, args=(name,))

if (submitted or st.session_state.pop("auto", False)) and brand_in.strip():
    out = analyze(brand_in, strict=not st.session_state.get("include_cmp", True))
    if out:
        st.session_state["result"] = out

res = st.session_state.get("result")
if res:
    render(res)
else:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    for col, (n, t, d) in zip((a, b, c), (
            ("01", "Collect", "Public YouTube comments from the top review videos for your search."),
            ("02", "Understand", "An AI pass keeps only comments about the product and sorts them into aspects."),
            ("03", "Report", "Counts and verdicts are computed in code. Every claim cites a comment."))):
        col.markdown(f'<div class="step"><div class="n">{n}</div><div class="t">{t}</div>'
                     f'<div class="d">{d}</div></div>', unsafe_allow_html=True)

with st.expander("Method and measured results"):
    st.markdown("""
**Pipeline:** YouTube comments, then a Gemini classifier for relevance and aspect, then a Gemini aspect-stance pass,
then counts and verdicts computed in code, then a report where every claim cites a comment id.

**Measured on hand-labeled data (small samples):**
- Fine-tuned RoBERTa vs the pretrained model on TweetEval: macro-F1 0.707 vs 0.725 (CI of the difference excludes zero). It did not help, so the pretrained model is used.
- Relevance filter (the deployed prompt) on 161 hand-labeled Nike comments: off-topic share among kept comments 14.9% to 2.7%, but only 78% of on-target comments kept. An earlier, looser prompt reached 4.4% with 96% kept.
- Aspect-stance pass vs pretrained RoBERTa on 98 Nike rows: negative recall 0.90 vs 0.64 (comment-level labels).

**Not validated:** the reports themselves have not been audited claim by claim.
""")
