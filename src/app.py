import streamlit as st
from pipeline import build_dataset
from insights import prep, generate_report

st.set_page_config(page_title="Brand Sentiment", layout="centered")
st.title("Brand sentiment from YouTube comments")


@st.cache_data(show_spinner=False)
def run(brand):
    return prep(build_dataset(brand))


brand = st.text_input("Brand or product", placeholder="e.g. Nike shoes, Mahindra Thar")

if st.button("Analyze") and brand.strip():
    try:
        with st.spinner("Collecting and classifying comments (a few minutes for a new brand)..."):
            df = run(brand.strip())
    except Exception as e:
        st.error(f"Failed (YouTube/Gemini quota or network?): {e}")
        st.stop()
    if not df["relevant"].any():
        st.warning("No on-topic English comments found. Try a more specific query.")
        st.stop()
    st.session_state["result"] = (brand.strip(), df)
    st.session_state.pop("report", None)

if "result" in st.session_state:
    b, df = st.session_state["result"]
    kept = df[df["relevant"] & df["sentiment"].notna()]
    share = (kept["sentiment"].value_counts(normalize=True)
             .reindex(["positive", "neutral", "negative"]).fillna(0).mul(100).round(1))

    st.subheader(f"{b}: {len(kept)} on-topic comments (of {len(df)} collected)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", f"{share['positive']}%")
    c2.metric("Neutral", f"{share['neutral']}%")
    c3.metric("Negative", f"{share['negative']}%")
    st.bar_chart(share)
    if len(kept) < 100:
        st.warning("Small sample: treat these percentages as rough.")

    for lab in ["positive", "negative"]:
        st.markdown(f"**Top {lab} comments**")
        top = kept[kept["sentiment"] == lab].sort_values(
            ["confidence", "likes"], ascending=False).head(3)
        for _, r in top.iterrows():
            st.write(f"- [{r['confidence']}] {str(r['text'])[:200]}")

    st.divider()
    if st.button("Generate strengths / weaknesses report"):
        with st.spinner("Analyzing comments..."):
            res = generate_report(b, df)
        st.session_state["report"] = (b,) + res

    if "report" in st.session_state and st.session_state["report"][0] == b:
        _, stats, qdf, text, notes = st.session_state["report"]
        if not stats.empty:
            st.markdown("**Aspect breakdown**")
            st.dataframe(stats.rename(columns={
                "pos_pct": "% positive", "neg_pct": "% negative", "net": "net score"}),
                hide_index=True)
        st.markdown(text)
        if notes:
            st.warning("Automatic checks flagged: " + "; ".join(notes))
        if not qdf.empty:
            with st.expander("Evidence comments cited by the report"):
                for r in qdf.itertuples():
                    st.write(f"**{r.id}** ({r.aspect}, {r.sentiment}): {r.text}")

    st.info(
        "Accuracy notes (single annotator). Sentiment model, macro-F1: 0.71 on Mahindra Thar "
        "(94 hand-labeled comments, 95% CI 0.60-0.80) and 0.63 on Nike (137 comments, 0.55-0.71). "
        "In both samples it caught only about 60% of negative comments, so the negative share "
        "is likely understated. Relevance filter: 4-6% of kept comments were still off-target in "
        "hand-labeled samples. Suggestions are hypotheses from commenters, not validated advice."
    )
