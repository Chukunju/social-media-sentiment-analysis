# Brand sentiment from YouTube comments

Type a brand or product and get sentiment shares, top comments, and an evidence-linked summary of recurring praise and complaints.

## How it works

1. Collect: YouTube Data API. Searches "<brand> review", pulls up to 100 top comments from each of 8 videos, keeps English comments.
2. Relevance and aspects: a Gemini classifier keeps comments that express an opinion about the brand or product and assigns each one an aspect (aspects are proposed per brand from a sample of its comments).
3. Sentiment: pretrained cardiffnlp/twitter-roberta-base-sentiment-latest (evaluated here, not trained by me). A TF-IDF + logistic regression baseline is trained on TweetEval for comparison.
4. Report: an aspect table (mention counts, % positive/negative, verdict from fixed thresholds computed in code) plus a Gemini-written summary that must cite numbered comments. Automatic checks confirm that cited ids exist, counts match the table, and sentiment tags match.

## Results

All hand-labeled sets use a single annotator (the author), so intervals are wide.

| Check | Data | Result |
|---|---|---|
| TF-IDF + LogReg baseline | TweetEval test, 12,284 tweets | macro-F1 0.587 |
| RoBERTa (pretrained) | TweetEval test | macro-F1 0.725 |
| RoBERTa vs baseline on YouTube | Nike, 137 hand-labeled comments | 0.633 (95% CI 0.55-0.71) vs 0.539 |
| RoBERTa on a held-out brand | Mahindra Thar, 94 hand-labeled comments | 0.712 (95% CI 0.60-0.80) |
| Negative-comment recall | Nike 30/50, Thar 11/19 | about 0.6 on both |
| Relevance filter, development brand | Nike, 161 labeled | off-target 14.9% (keyword filter) to 4.3% (classifier); 96.4% of on-topic comments kept |
| Relevance filter, held-out | Thar, 100 kept comments | 6.0% off-target (95% CI 3-12%), lenient definition |
| Aspect assignment | Thar, 60 labeled | right for 79% of genuine product comments (95% CI 66-88%); 20% of sampled comments were questions or reviewer chatter that should not have received an aspect |
| Report citations | 2 generated reports, AI-assisted review | 8 of 28 praise/complaint clauses had at least one weak or off-target citation |

## Limitations

- YouTube commenters are not a representative sample of customers, and a brand query mixes several products.
- Sentiment is scored per comment, not per aspect.
- The sentiment model catches only about 60% of negative comments, so negative share and weaknesses are likely understated.
- Questions and reviewer or channel chatter still leak into aspect counts.
- Upgrade suggestions are hypotheses drawn from a few comments and are not validated. They are only generated for non-borderline weakness aspects.
- Not tested on Hinglish or non-English comments.
- LLM free-tier limits and model names change over time.

## Run it

Create a .env file with YOUTUBE_API_KEY and GEMINI_API_KEY, then:

    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python -m streamlit run src\app.py

Scripts in src: download_data.py, train_baseline.py, eval_roberta.py (TweetEval), collector.py, classify.py, pipeline.py (data), eval_*.py and label_*.py (evaluation and labeling), insights.py and report_check.py (reports).
