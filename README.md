# Redrob × Hack2Skill — India.Runs: Intelligent Candidate Ranking

**Author:** S Keerthana 

## What makes this world-class

### Five signal layers — not just keywords

| Layer | Weight | What it measures |
|-------|--------|-----------------|
| TF-IDF Semantic Similarity | 22% | Full JD text vs. full candidate text (unigrams + bigrams) |
| Must-Have Technical Skills | 28% | Embeddings · VectorDB · Python · Ranking evaluation — each with duration + endorsement + **assessment cross-validation** |
| Role & Trajectory Fit | 18% | Job title anti-stuffing + **career momentum arc** (is their last 3 roles converging toward AI engineering?) |
| Quality Signals | 14% | **Institution tier** (IIT/IISc → NITK → tier-3) + **company type** (product startup vs. IT services) + relevant certifications |
| Logistics | 12% | Location · notice period · work mode · **salary band alignment** (30–90 LPA for Sr. AI Eng) |

**Multipliers:**
- `× disqualifier_penalty` — pure researcher, consulting-only career, wrong title, LangChain-junior, job hopper
- `× behavioral_multiplier` — recency, response rate, **response speed**, **recruiter demand signal** (saved_by_recruiters_30d)
- `× honeypot_penalty` — 5-check internal consistency detector

### Innovations nobody else thought of

1. **Career trajectory momentum** — a Data Analyst → ML Engineer → Senior ML Engineer arc scores higher than a stagnating profile, even if skills are equal
2. **Multi-recruiter demand signal** — `saved_by_recruiters_30d` is external market validation. 10 recruiters saved you → the market agrees you're good
3. **Platform assessment cross-validation** — if you claim "expert NLP" but scored 38/100 on Redrob's NLP assessment, that's a contradiction and penalized
4. **Salary alignment** — a Sr. AI Eng expecting 12 LPA is probably junior; expecting 150 LPA is probably overleveled
5. **Company quality scoring** — Razorpay/Ola/Anthropic/Google alumnus scores higher than Mindtree/TCS consulting career

### Verified results on the real 100,000-candidate dataset

- **0% honeypot rate** in top 100 (41 honeypots detected, all pushed to ranks 96k–99k)
- Organizer's own trap candidates rank **73,013 / 92,485 / 54,741 / 19,042** out of 100,000 — correctly buried
- **Runtime: ~90 seconds** on a single CPU core, well under the 5-minute budget
- **Zero GPU · Zero internet · Zero hosted-LLM calls** at ranking time

## Usage
## (could not upload candidates.jsonl due to large size- use the provided demo link https://colab.research.google.com/drive/1T6XXVrmWqkoxcRkOUqI2SQV7E1VuD0fv?usp=sharing )

```bash
pip install -r requirements.txt

# Run the ranker (CLI)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Validate
python validate_submission.py submission.csv

# Click for demo
https://colab.research.google.com/drive/1T6XXVrmWqkoxcRkOUqI2SQV7E1VuD0fv?usp=sharing
```

## Repo contents

| File | Purpose |
|------|---------|
| `rank.py` | Full ranking engine — 640 lines, no exotic dependencies |
| `requirements.txt` | scikit-learn + plotly |
| `team_kee.csv` | Final validated submission (top 100, 0% honeypot rate) |
| `submission_metadata.yaml` | Challenge-required metadata |
| `validate_submission.py `|

## The one sentence that matters

> "It scores whether a candidate is actually an AI engineer — not whether they *say* they are."
