# Nike shoes

             aspect  mentions  pos_pct  neg_pct  net  verdict  neg_n  borderline
             design       127       51       19   32 strength     24       False
        performance        80       46        8   38 strength      6       False
    brand / company        79       49       22   27 strength     17       False
     sizing and fit        56       23       43  -20 weakness     24       False
            comfort        48       40       29   11    mixed     14       False
      price / value        34       32       29    3    mixed     10       False
         durability        15       40       27   13    mixed      4       False

### Strengths
* **design** - 127 mentions (51% positive, 19% negative). Praise: Toebox shape looks great and quality materials were used [Q7]. Complaints: One commenter dislikes the front design and pink color [Q2], while others note a too-long tongue [Q3], a borrowed mold from the KD 15 [Q4], or awkward spacing [Q1].
* **performance** - 80 mentions (46% positive, 8% negative). Praise: Commenters call them great performers and among the best basketball shoe purchases ever [Q12, Q13]. Complaints: One commenter missed layups because the shoe was unexpectedly light [Q10].
* **brand / company** - 79 mentions (49% positive, 22% negative). Praise: Commenters express excitement and call them a hyped, best sneaker release [Q19, Q20, Q21]. Complaints: Commenters criticize Nike's claims, pricing, and lack of adequate research [Q15, Q16, Q18].

### Weaknesses
* **sizing and fit** - 56 mentions (23% positive, 43% negative). Praise: Commenters note the fit is great for their feet [Q27]. Complaints: Commenters state Nike shoes are not wide-foot friendly and feature tight, tapered, and pointed toe boxes [Q23, Q24, Q25].

### Mixed
* **comfort** - 48 mentions (40% positive, 29% negative). Praise: Commenters highlight high arch support from the shank and comfortable wear [Q34, Q35]. Complaints: One commenter dislikes thick heel padding [Q30], while others complain about triangular toe boxes and shoes hurting their feet [Q31, Q32].
* **price / value** - 34 mentions (32% positive, 29% negative). Praise: Commenters feel the shoes are worth every dollar and reasonably priced [Q41]. Complaints: Commenters describe the shoes as a waste of money, overpriced for cushlon foam, and subject to an unwarranted price hike [Q36, Q37, Q39].
* **durability** - 15 mentions (40% positive, 27% negative). Praise: Commenters cite durable outsoles and suitability for outdoor hooping [Q45, Q46]. Complaints: Commenters raise concerns about dirt, mud, and waste getting stuck in node holes [Q43, Q44].

### Suggested upgrades
* **sizing and fit**: Observed: Commenters state Nike shoes are not wide-foot friendly and feature tight, tapered, and pointed toe boxes [Q23, Q24]. Hypothesis (inference, not from commenters): Broaden the toebox shape and offer dedicated wide-width options to accommodate diverse foot structures.

### Limits
- Based on 463 on-topic comments out of 634 collected; YouTube commenters are not a representative customer sample.
- Sentiment is scored per comment, not per aspect, and the LLM's assignment of comments to aspects has not been validated.
- The sentiment model caught only about 60% of negative comments in hand-labeled tests, so weaknesses are likely understated.
- Verdicts use fixed thresholds: strength net +20 or more, weakness net -10 or less, thin under 10 mentions; aspects within 5 points of a threshold are marked borderline and get no upgrade suggestions.
- Format checks (ids exist, counts match, sentiment tags match) are automatic. Whether a cited comment truly supports a claim is not verified, and a manual audit found some off-target citations.

Format checks: passed (claim support not verified)
