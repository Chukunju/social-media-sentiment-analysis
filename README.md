
# What do people really say about a product?

Type any brand or product. The app reads YouTube comments, keeps the ones about the product itself, scores sentiment per aspect (design, price, battery and so on), and writes a report where every claim points to a real comment.

## How it works

1. **Collect.** Top review videos for the search (8 videos, up to 100 top comments each, English only).
2. **Check the videos.** Optional strict mode drops videos whose titles compare several products.
3. **Relevance and aspect.** A Gemini classifier keeps comments that are about the product itself and assigns each to one aspect. The aspect list is proposed per product, so it fits a laptop or a car.
4. **Aspect-level stance.** A second Gemini pass judges positive, negative or neutral toward that aspect, and flags comments that are really about something else.
5. **Report.** Counts, thin-evidence flags and strength or weakness verdicts are computed in code. The model only writes short summaries, and every comment id it cites is checked against the quotes it was given.

## Measured results

All numbers come from small hand-labeled samples. Treat them as indications, not guarantees.

| Check | Result |
|---|---|
| Fine-tuned RoBERTa vs pretrained model, TweetEval test set (about 12k tweets) | Macro-F1 0.707 vs 0.725; difference -0.018, 95% CI [-0.024, -0.011]. Fine-tuning did not help, so the pretrained model is used. |
| Same comparison on hand-labeled YouTube comments | Nike (n=137): 0.598 vs 0.633, CI of the difference [-0.117, +0.040]. Thar (n=94): 0.634 vs 0.712, CI [-0.164, +0.002]. Neither interval excludes zero. Negative recall was lower for the fine-tuned model on all three sets. |
| Relevance filter on 161 hand-labeled Nike comments (deployed prompt) | Off-topic share among kept comments fell from 14.9% to 2.7%. It removed 21 of 24 off-topic comments but kept only 78% of on-target ones. An earlier, looser prompt reached 4.4% off-topic with 96% kept. All 161 comments had already passed an older keyword filter, so comments it wrongly dropped are not measured. |
| Aspect-level stance vs pretrained RoBERTa, 98 Nike rows | Macro-F1 0.724 vs 0.669; negative recall 0.90 vs 0.64. Hand labels are comment-level while the stance is aspect-level, so this is a rough comparison. |

## Limitations

- A brand search can cover several models. Strict mode drops comparison videos, but leaves fewer comments.
- Some off-topic or other-model comments still pass the filters. Spot checks of generated reports found cited comments about other models, about the video itself, or misread. The reports have not been audited claim by claim.
- Relevance and stance were evaluated on one brand (Nike) only, with small samples.
- Suggestions are hypotheses drawn from a small set of quoted comments, not validated advice.
- The public demo limits how many new analyses can run. Brands that were already analyzed load instantly.

## Run locally


Add GEMINI_API_KEY and YOUTUBE_API_KEY to a .env file, then:

