import sys


def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            sys.exit(f"STOP: expected 1 match in {path}, found {n}, for: {old[:70]!r}. Nothing written for this file.")
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)
    print("patched", path)


patch("src/insights.py", [
    (r'''COLS = ["aspect", "mentions", "pos_pct", "neg_pct", "net", "verdict"]''',
     r'''COLS = ["aspect", "mentions", "pos_pct", "neg_pct", "net", "verdict", "neg_n", "borderline"]'''),
    (r'''def aspect_stats(df):''',
     r'''def near_threshold(n, net):
    v = verdict(n, net)
    return bool((v == "strength" and net - STRENGTH_NET < 5)
                or (v == "weakness" and WEAKNESS_NET - net < 5))


def aspect_stats(df):'''),
    (r'''"net": pos - neg, "verdict": verdict(n, pos - neg)})''',
     r'''"net": pos - neg, "verdict": verdict(n, pos - neg),
                     "neg_n": int((g["sentiment"] == "negative").sum()),
                     "borderline": near_threshold(n, pos - neg)})'''),
    (r'''g = g[(txt.str.split().str.len() >= 8) & ~txt.str.contains(REVIEWER_PRAISE)]''',
     r'''g = g[(txt.str.split().str.len() >= 8) & ~txt.apply(lambda s: bool(REVIEWER_PRAISE.search(s)))]
        g = g.loc[~g["text"].astype(str).str.lower().str[-30:].duplicated()]'''),
    (r'''elig = stats.loc[stats["verdict"] == "weakness", "aspect"].tolist()''',
     r'''elig = stats.loc[(stats["verdict"] == "weakness") & ~stats["borderline"], "aspect"].tolist()'''),
    (r'''elig = {a.lower() for a in stats.loc[stats["verdict"] == "weakness", "aspect"]}''',
     r'''elig = {a.lower() for a in stats.loc[(stats["verdict"] == "weakness") & ~stats["borderline"], "aspect"]}'''),
    (r'''other = [a for a in stats["aspect"] if a.lower() not in elig and a.lower() in sec.lower()]''',
     r'''other = [a for a in stats["aspect"] if a.lower() not in elig and re.search(r"\b" + re.escape(a.lower()) + r"\b", sec.lower())]'''),
    (r'''- Copy N, P and Q exactly from the table. If a detail appears in only one comment, say "one commenter".''',
     r'''- Copy N, P and Q exactly from the table. If a detail appears in only one comment, say "one commenter".
- If an aspect's borderline column is True, write the word borderline right after its name.
- Do not cite a comment that is about a different brand or model, a reseller, or the reviewer, or that does not itself address the aspect (for example, praise for a video intro is not evidence about styling).'''),
    (r'''f"- Verdicts use fixed thresholds: strength net +{STRENGTH_NET} or more, weakness net {WEAKNESS_NET} or less, thin under {MIN_SOLID} mentions.",''',
     r'''f"- Verdicts use fixed thresholds: strength net +{STRENGTH_NET} or more, weakness net {WEAKNESS_NET} or less, thin under {MIN_SOLID} mentions; aspects within 5 points of a threshold are marked borderline and get no upgrade suggestions.",
        "- Format checks (ids exist, counts match, sentiment tags match) are automatic. Whether a cited comment truly supports a claim is not verified, and a manual audit found some off-target citations.",'''),
])

patch("src/report_check.py", [
    (r'''print("\nAutomatic checks:", notes or "passed")''',
     r'''print("\nFormat checks:", notes or "passed (ids, counts, tags only; claim support is not verified)")'''),
    (r'''Automatic checks: {notes or 'passed'}''',
     r'''Format checks: {notes or 'passed (claim support not verified)'}'''),
])
