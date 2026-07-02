#!/usr/bin/env python3
"""
Redrob × Hack2Skill — India.Runs Intelligent Candidate Ranking
Author: S Keerthana 
================================================================
World-class, fully offline, CPU-only candidate ranker for
100,000 profiles against a Senior AI Engineer JD.

ARCHITECTURE — 5 Signal Layers + 3 Multipliers:
  Layer 1: Technical Fit        (TF-IDF semantic + must-have skills)
  Layer 2: Role & Trajectory    (title fit + career arc momentum)
  Layer 3: Quality Signals      (institution tier + platform assessments + certs)
  Layer 4: Behavioral & Market  (recency + response + recruiter demand)
  Layer 5: Logistics            (location + notice + salary + work mode)
  × Disqualifier penalty stack
  × Honeypot consistency detector
  × Behavioral multiplier

INNOVATIONS vs. standard approaches:
  • Career trajectory momentum: is their arc *heading toward* this role?
  • Multi-recruiter demand signal: external market validation
  • Platform-assessment cross-validation: self-claim vs. tested score
  • Salary alignment: senior AI eng band awareness
  • Certification relevance scoring
  • Response-speed engagement signal

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse, csv, json, math, re, sys, time
from datetime import date, datetime
from sklearn.feature_extraction.text import TfidfVectorizer

TODAY = date(2026, 6, 29)

# ─────────────────────────────────────────────────────────────
# JD Knowledge Base (encoded from the actual JD text)
# ─────────────────────────────────────────────────────────────
JD_TEXT = """
Senior AI Engineer Founding Team Redrob AI. Own intelligence layer ranking
retrieval matching systems candidate JD matching scale. Production experience
embeddings retrieval sentence transformers openai embeddings BGE E5 deployed
real users. Handled embedding drift index refresh retrieval quality regression
production. Vector databases hybrid search Pinecone Weaviate Qdrant Milvus
OpenSearch Elasticsearch FAISS. Strong Python code quality. Evaluation
frameworks ranking systems NDCG MRR MAP offline online correlation AB test.
LLM fine tuning LoRA QLoRA PEFT. Learning to rank XGBoost neural ranking.
HR tech recruiting marketplace products. Distributed systems large scale
inference optimization model serving. Open source contributions AI ML.
Shipped end-to-end ranking search recommendation system real users meaningful
scale. Hybrid retrieval dense sparse. NLP information retrieval semantic
search recommendation.
"""

# Must-have skill groups (each checked independently)
MUST_EMBED   = ["sentence-transformer","sentence transformer","openai embedding",
                "bge","e5","text-embedding","cohere embed","instructor embedding","embedding"]
MUST_VECTORDB = ["pinecone","weaviate","qdrant","milvus","opensearch",
                 "elasticsearch","faiss","vespa","pgvector","vector database",
                 "vector db","vector search","hybrid search","hybrid retrieval"]
MUST_PYTHON  = ["python"]
MUST_EVAL    = ["ndcg","mrr","map","a/b test","ab test","offline evaluation",
                "online evaluation","evaluation framework","learning to rank",
                "learning-to-rank","ltr","precision@","recall@"]

# Nice-to-have skill groups
NICE_FINETUNE  = ["lora","qlora","peft","fine-tuning","finetuning","fine tune"]
NICE_LTR       = ["xgboost","lightgbm","catboost","learning-to-rank","ranknet","lambdamart"]
NICE_HRTECH    = ["recruiting","hr-tech","hrtech","ats","talent acquisition",
                  "recruitment","job board","marketplace","hiring platform"]
NICE_OSS       = ["open source","github","open-source","contributor","maintainer"]
NICE_DIST      = ["kubernetes","spark","kafka","distributed","model serving","triton","vllm","ray"]

# Title signals
POSITIVE_TITLES = ["machine learning engineer","ml engineer","ai engineer",
                   "applied scientist","research engineer","data scientist",
                   "nlp engineer","search engineer","search relevance",
                   "ranking engineer","recommender","recommendation",
                   "founding engineer","staff engineer","principal engineer",
                   "ml platform","ai platform","deep learning engineer",
                   "information retrieval","senior engineer","backend engineer",
                   "software engineer","data engineer","platform engineer"]
NEGATIVE_TITLES = ["marketing manager","hr manager","human resources",
                   "content writer","content strategist","sales","recruiter",
                   "talent acquisition specialist","business analyst",
                   "graphic designer","accountant","financial analyst",
                   "operations manager","social media","copywriter",
                   "customer success","administrative","hr business partner",
                   "payroll","office manager","qa tester","quality analyst"]

NLP_IR_KEYWORDS = ["nlp","natural language","retrieval","search","ranking",
                   "embedding","information retrieval","semantic search",
                   "transformer","llm","rag","recommendation","recommender"]
CV_ONLY_KEYWORDS = ["computer vision","image classification","object detection",
                    "speech recognition","robotics","autonomous","slam","lidar"]
PRODUCTION_KEYWORDS = ["production","deployed","shipped","real users","at scale",
                       "live system","latency","uptime","rollout","million users"]

CONSULTING_FIRMS = ["tcs","tata consultancy","infosys","wipro","accenture",
                    "cognizant","capgemini","hcl","tech mahindra","mindtree",
                    "mphasis","ltimindtree","l&t infotech"]
TOP_PRODUCT_SIGNALS = ["openai","google","meta","amazon","microsoft","apple",
                       "netflix","uber","airbnb","stripe","figma","anthropic",
                       "cohere","hugging face","databricks","snowflake","atlassian",
                       "razorpay","zepto","meesho","cred","groww","navi","slice",
                       "swiggy","zomato","flipkart","ola","paytm","byju"]

RELEVANT_CERTS = ["aws certified machine learning","aws ml","gcp professional ml",
                  "tensorflow developer","tensorflow certificate",
                  "deep learning specialization","machine learning specialization",
                  "mlops","hugging face","coursera ml","fast.ai","pytorch",
                  "microsoft certified ai","azure ai","databricks certified"]

PREFERRED_CITIES = ["pune","noida"]
TIER1_CITIES    = ["bengaluru","bangalore","hyderabad","mumbai","delhi",
                   "gurugram","gurgaon","ncr","chennai"]

# Senior AI Eng at a product startup — expected salary band (LPA)
SALARY_BAND_MIN = 30.0
SALARY_BAND_MAX = 90.0

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def parse_date(s):
    if not s: return None
    try: return datetime.strptime(s, "%Y-%m-%d").date()
    except: return None

def sl(s): return (s or "").lower()

def any_kw(text, kws): return any(k in text for k in kws)
def count_kw(text, kws): return sum(1 for k in kws if k in text)

def text_blob(cand):
    parts = [cand["profile"].get("headline",""),
             cand["profile"].get("summary",""),
             cand["profile"].get("current_title",""),
             cand["profile"].get("current_industry","")]
    for ch in cand.get("career_history",[]):
        parts += [ch.get("title",""), ch.get("description",""), ch.get("industry","")]
    for sk in cand.get("skills",[]):
        n = sk.get("name","")
        parts.append((n+" ")*max(1, min(3, (sk.get("endorsements",0)//5)+1)))
    for ct in cand.get("certifications",[]):
        parts.append(ct.get("name","") + " " + ct.get("issuer",""))
    return " ".join(parts)

# ─────────────────────────────────────────────────────────────
# LAYER 1: Technical Fit
# ─────────────────────────────────────────────────────────────

def skill_trust(skills_by_name, assessed_lower, full_text, kw_list):
    """Score a skill group with duration + endorsement + assessment cross-check."""
    best = 0.0; evidence = None
    for name, s in skills_by_name.items():
        if not any(k in name for k in kw_list): continue
        prof_w = {"beginner":0.35,"intermediate":0.60,"advanced":0.85,"expert":1.0}.get(
            s.get("proficiency","beginner"), 0.35)
        dur = s.get("duration_months",0) or 0
        trust = min(1.0, dur/12.0) if dur > 0 else 0.05
        endorse_bonus = min(0.15, math.log1p(s.get("endorsements",0))/20)

        # Platform assessment cross-validation (INNOVATION)
        assess_bonus = 0.0
        assess_penalty = 0.0
        for an, av in assessed_lower.items():
            if an in name or name in an:
                if av >= 70: assess_bonus = 0.30
                elif av >= 50: assess_bonus = 0.10
                elif av < 40 and s.get("proficiency") == "expert":
                    assess_penalty = 0.30  # claimed expert, failed test
                break

        val = min(1.0, prof_w * trust + endorse_bonus + assess_bonus - assess_penalty)
        if val > best: best = val; evidence = (s.get("name"), s.get("proficiency"), dur, assess_bonus)

    if best == 0.0 and any_kw(full_text, kw_list): best = 0.18
    return best, evidence

def compute_technical_fit(cand, full_text):
    skills_by_name = {sl(s.get("name","")): s for s in cand.get("skills",[])}
    sig = cand.get("redrob_signals",{}) or {}
    assessed_lower = {k.lower():v for k,v in (sig.get("skill_assessment_scores") or {}).items()}

    e_s, e_e = skill_trust(skills_by_name, assessed_lower, full_text, MUST_EMBED)
    v_s, v_e = skill_trust(skills_by_name, assessed_lower, full_text, MUST_VECTORDB)
    p_s, p_e = skill_trust(skills_by_name, assessed_lower, full_text, MUST_PYTHON)
    m_s, m_e = skill_trust(skills_by_name, assessed_lower, full_text, MUST_EVAL)

    must_have = (e_s + v_s + p_s + m_s) / 4.0

    ft_s, _ = skill_trust(skills_by_name, assessed_lower, full_text, NICE_FINETUNE)
    lt_s, _ = skill_trust(skills_by_name, assessed_lower, full_text, NICE_LTR)
    hr_s, _ = skill_trust(skills_by_name, assessed_lower, full_text, NICE_HRTECH)
    os_s, _ = skill_trust(skills_by_name, assessed_lower, full_text, NICE_OSS)
    ds_s, _ = skill_trust(skills_by_name, assessed_lower, full_text, NICE_DIST)
    nice = min(1.0, (ft_s + lt_s + hr_s + os_s + ds_s) / 5.0)

    return must_have, nice, (e_e, v_e, p_e, m_e)

# ─────────────────────────────────────────────────────────────
# LAYER 2: Role & Trajectory Fit
# ─────────────────────────────────────────────────────────────

def compute_role_trajectory(cand, full_text):
    career = cand.get("career_history",[]) or []
    prof = cand["profile"]

    # Current title scoring (anti-keyword-stuffer)
    all_titles = sl(prof.get("current_title","")) + " " + \
                 " ".join(sl(c.get("title","")) for c in career)
    pos_hits = count_kw(all_titles, POSITIVE_TITLES)
    neg_hits = count_kw(all_titles, NEGATIVE_TITLES)

    if neg_hits > 0 and pos_hits == 0: title_fit = 0.03
    elif neg_hits > 0 and pos_hits > 0: title_fit = 0.20
    elif pos_hits >= 2: title_fit = 1.0
    elif pos_hits == 1: title_fit = 0.70
    else: title_fit = 0.12

    is_stuffer = (neg_hits > 0 and pos_hits == 0)

    # INNOVATION: Career trajectory momentum
    # Sort career by date, look at LAST 3 roles, check if converging on ML/AI
    sorted_career = sorted(career,
        key=lambda c: parse_date(c.get("start_date")) or date(2000,1,1))

    recent = sorted_career[-3:] if len(sorted_career) >= 3 else sorted_career
    if not recent:
        trajectory = 0.5
    else:
        # Score each recent role for ML/AI relevance
        role_scores = []
        for c in recent:
            rt = sl(c.get("title","")) + " " + sl(c.get("description",""))
            nlp_hits = count_kw(rt, NLP_IR_KEYWORDS)
            pos_t    = count_kw(rt, POSITIVE_TITLES)
            role_scores.append(min(1.0, (nlp_hits * 0.15 + pos_t * 0.30)))

        if len(role_scores) >= 2:
            # Is the arc going UP (improving), FLAT, or DOWN (drifting away)?
            trend = role_scores[-1] - role_scores[0]
            base  = sum(role_scores) / len(role_scores)
            trajectory = min(1.0, base + 0.3 * trend)
        else:
            trajectory = role_scores[0] if role_scores else 0.3

    role_trajectory = 0.65 * title_fit + 0.35 * trajectory
    return role_trajectory, title_fit, trajectory, is_stuffer

# ─────────────────────────────────────────────────────────────
# LAYER 3: Quality Signals
# ─────────────────────────────────────────────────────────────

def compute_quality_signals(cand, full_text):
    # Education tier (INNOVATION: unused before)
    edu = cand.get("education",[]) or []
    best_tier = max(
        {"tier_1":1.0,"tier_2":0.65,"tier_3":0.35,"tier_4":0.20}.get(
            e.get("tier","tier_4"), 0.20)
        for e in edu) if edu else 0.20
    edu_score = 0.35 + 0.65 * best_tier  # tier_1 → 1.0, tier_4 → 0.55

    # Company quality: product co > consulting (INNOVATION: company_size + known names)
    career = cand.get("career_history",[]) or []
    company_scores = []
    for c in career:
        comp = sl(c.get("company",""))
        is_consulting = any(f in comp for f in CONSULTING_FIRMS)
        is_top_product = any(f in comp for f in TOP_PRODUCT_SIGNALS)
        size = c.get("company_size","") or ""
        # startup (11-500) at product role = high; big IT services = low
        is_startup_size = size in ("11-50","51-200","201-500")
        is_it_giant = size == "10001+" and is_consulting

        if is_top_product: cs = 1.0
        elif is_startup_size and not is_consulting: cs = 0.80
        elif is_consulting: cs = 0.25
        elif is_it_giant: cs = 0.15
        else: cs = 0.50
        company_scores.append(cs)
    company_score = sum(company_scores)/len(company_scores) if company_scores else 0.40

    # Certification relevance (INNOVATION)
    certs = cand.get("certifications",[]) or []
    cert_text = " ".join(sl(ct.get("name","")) + " " + sl(ct.get("issuer","")) for ct in certs)
    cert_hits = count_kw(cert_text, RELEVANT_CERTS)
    cert_score = min(1.0, 0.4 + cert_hits * 0.25)

    quality = 0.40 * company_score + 0.35 * edu_score + 0.25 * cert_score
    return quality, edu_score, company_score, cert_score

# ─────────────────────────────────────────────────────────────
# LAYER 4: Behavioral & Market Signals
# ─────────────────────────────────────────────────────────────

def compute_behavioral_market(cand):
    sig = cand.get("redrob_signals",{}) or {}

    # Recency / engagement
    last_active = parse_date(sig.get("last_active_date"))
    days_inactive = (TODAY - last_active).days if last_active else 999
    if days_inactive <= 14: recency = 1.0
    elif days_inactive <= 45: recency = 0.85
    elif days_inactive <= 90: recency = 0.65
    elif days_inactive <= 180: recency = 0.45
    else: recency = 0.25

    resp_rate = sig.get("recruiter_response_rate", 0.3) or 0.0
    open_flag = 1.0 if sig.get("open_to_work_flag") else 0.50

    # INNOVATION: Response speed signal
    avg_resp_h = sig.get("avg_response_time_hours", 72) or 72
    if avg_resp_h <= 4: resp_speed = 1.0
    elif avg_resp_h <= 12: resp_speed = 0.85
    elif avg_resp_h <= 24: resp_speed = 0.70
    elif avg_resp_h <= 72: resp_speed = 0.50
    else: resp_speed = 0.25

    # INNOVATION: Multi-recruiter demand signal
    # Sweet spot: 3-20 saves = desirable but not already committed
    saved = sig.get("saved_by_recruiters_30d", 0) or 0
    if 3 <= saved <= 20: demand = 1.0
    elif saved > 20: demand = 0.75  # very hot = may be in late-stage elsewhere
    elif saved == 1 or saved == 2: demand = 0.70
    else: demand = 0.45

    # Platform trust
    verified = (1 if sig.get("verified_email") else 0) + \
               (1 if sig.get("verified_phone") else 0) + \
               (1 if sig.get("linkedin_connected") else 0)
    trust = 0.65 + 0.117 * verified

    completeness = (sig.get("profile_completeness_score", 50) or 50) / 100.0
    interview_comp = sig.get("interview_completion_rate", 0.5) or 0.5

    behavioral = (0.22*recency + 0.18*resp_rate + 0.15*resp_speed +
                  0.18*demand + 0.12*open_flag + 0.08*trust + 0.07*completeness)
    multiplier = 0.50 + 0.70 * behavioral  # range ~[0.5, 1.2]

    return multiplier, {
        "recency": recency, "resp_rate": resp_rate, "resp_speed": resp_speed,
        "demand": demand, "days_inactive": days_inactive, "saved": saved
    }

# ─────────────────────────────────────────────────────────────
# LAYER 5: Logistics
# ─────────────────────────────────────────────────────────────

def compute_logistics(cand):
    prof = cand["profile"]
    sig = cand.get("redrob_signals",{}) or {}

    loc = sl(prof.get("location",""))
    country = sl(prof.get("country",""))
    relocate = bool(sig.get("willing_to_relocate", False))
    work_mode = sl(sig.get("preferred_work_mode",""))

    if country not in ("india","in",""):
        location_fit = 0.60 if relocate else 0.15
    elif any(c in loc for c in PREFERRED_CITIES): location_fit = 1.0
    elif any(c in loc for c in TIER1_CITIES): location_fit = 0.85
    elif relocate: location_fit = 0.65
    else: location_fit = 0.35

    notice = sig.get("notice_period_days", 60) or 60
    if notice <= 15: notice_fit = 1.0
    elif notice <= 30: notice_fit = 0.90
    elif notice <= 60: notice_fit = 0.70
    elif notice <= 90: notice_fit = 0.45
    else: notice_fit = 0.25

    # Work mode: Redrob is a startup, likely onsite/hybrid preferred
    mode_fit = {"onsite":1.0,"hybrid":0.90,"flexible":0.80,"remote":0.55}.get(work_mode, 0.70)

    # INNOVATION: Salary alignment
    sal = sig.get("expected_salary_range_inr_lpa") or {}
    sal_min = sal.get("min", 20.0)
    sal_max = sal.get("max", 40.0)
    sal_mid = (sal_min + sal_max) / 2.0
    # Band: 30-90 LPA for senior AI eng at a startup
    if SALARY_BAND_MIN <= sal_mid <= SALARY_BAND_MAX: salary_fit = 1.0
    elif sal_mid < SALARY_BAND_MIN:
        # Under-expecting → may be junior, but not a dealbreaker
        salary_fit = 0.65 + 0.35 * (sal_mid / SALARY_BAND_MIN)
    else:
        # Over-expecting → likely too senior/expensive for this stage
        salary_fit = max(0.30, 1.0 - (sal_mid - SALARY_BAND_MAX) / SALARY_BAND_MAX)

    yoe = prof.get("years_of_experience", 0) or 0
    if 4 <= yoe <= 9: exp_fit = 1.0
    elif 2 <= yoe < 4 or 9 < yoe <= 13: exp_fit = 0.65
    elif yoe >= 14: exp_fit = 0.40
    else: exp_fit = 0.25

    logistics = (0.35*location_fit + 0.22*notice_fit + 0.15*mode_fit +
                 0.15*salary_fit + 0.13*exp_fit)
    return logistics, {"location":location_fit,"notice":notice_fit,"salary":salary_fit,"exp":exp_fit}

# ─────────────────────────────────────────────────────────────
# Disqualifier Penalty Stack
# ─────────────────────────────────────────────────────────────

def compute_penalty(cand, full_text, is_stuffer):
    career = cand.get("career_history",[]) or []
    prof = cand["profile"]
    flags = []
    penalty = 1.0

    companies = " ".join(sl(c.get("company","")) for c in career)
    industries = " ".join(sl(c.get("industry","")) for c in career)

    # Pure research / academic, no production
    research_like = any(kw in industries + companies for kw in
        ["university","research lab","academia","phd program","research institute"])
    if research_like and not any_kw(full_text, PRODUCTION_KEYWORDS) and len(career) > 0:
        penalty *= 0.30; flags.append("pure-research, zero production evidence")

    # Junior LangChain-only with < 3 years
    yoe = prof.get("years_of_experience", 0) or 0
    recent_only = all((c.get("duration_months",0) or 0) <= 14 for c in career) and len(career) <= 1
    if recent_only and "langchain" in full_text and yoe < 3:
        penalty *= 0.35; flags.append("recent LangChain-only, no deep ML history")

    # Leadership title > 18 months (away from hands-on code)
    cur = sl(prof.get("current_title",""))
    is_current_long = any(c.get("is_current") and (c.get("duration_months",0) or 0) > 18 for c in career)
    if any(t in cur for t in ["engineering manager","director","vp ","head of","chief"]) and is_current_long:
        penalty *= 0.55; flags.append("18mo+ non-coding leadership role")

    # Entire career at IT services/consulting
    if len(career) > 0 and all(any(f in sl(c.get("company","")) for f in CONSULTING_FIRMS) for c in career):
        penalty *= 0.45; flags.append("100% consulting career, no product company")

    # CV/speech-only ML without NLP/IR
    cv_heavy = count_kw(full_text, CV_ONLY_KEYWORDS)
    nlp_heavy = count_kw(full_text, NLP_IR_KEYWORDS)
    if cv_heavy >= 2 and nlp_heavy == 0:
        penalty *= 0.40; flags.append("CV/speech/robotics-only, no NLP/IR/search experience")

    # Keyword stuffer (HR Manager, Content Writer with AI skills)
    if is_stuffer:
        penalty *= 0.04
        flags.append("KEYWORD STUFFER: non-engineering title with AI skill claims")

    # Job hopper (3+ short stints < 12 months)
    short_stints = sum(1 for c in career if (c.get("duration_months",0) or 0) < 12)
    if short_stints >= 3 and len(career) >= 4:
        penalty *= 0.70; flags.append(f"{short_stints} stints < 12 months: high churn risk")

    return penalty, flags

# ─────────────────────────────────────────────────────────────
# Honeypot Detector
# ─────────────────────────────────────────────────────────────

def compute_honeypot(cand):
    career = cand.get("career_history",[]) or []
    skills = cand.get("skills",[]) or []
    edu    = cand.get("education",[]) or []
    prof   = cand["profile"]
    hp = 0; flags = []

    # 1. Expert with ~0 months duration
    expert_zero = sum(1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months",0) or 0) <= 1)
    if expert_zero >= 1: hp += 1; flags.append(f"{expert_zero} 'expert' skills with 0 months")
    if expert_zero >= 3: hp += 1

    # 2. Total career months >> stated years_of_experience
    yoe = prof.get("years_of_experience", 0) or 0
    total_career_months = sum((c.get("duration_months",0) or 0) for c in career)
    if total_career_months > (yoe * 12) + 24:
        hp += 1; flags.append("career months exceed stated YOE by 2+ years")

    # 3. YOE impossible given education start year
    earliest_edu = min((e.get("start_year", 9999) for e in edu), default=9999)
    if earliest_edu < 9999 and yoe > (2026 - earliest_edu) + 2:
        hp += 1; flags.append("claimed YOE exceeds possible working years since graduation")

    # 4. Date math mismatches in career roles
    for c in career:
        sd, ed = parse_date(c.get("start_date")), parse_date(c.get("end_date")) or TODAY
        if sd and ed:
            real_months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            if abs(real_months - (c.get("duration_months",0) or 0)) > 8:
                hp += 1; flags.append("career date/duration mismatch"); break

    # 5. Overlapping job dates
    dated_roles = [(parse_date(c.get("start_date")), parse_date(c.get("end_date")) or TODAY)
                   for c in career if parse_date(c.get("start_date"))]
    dated_roles.sort(key=lambda x: x[0])
    for i in range(len(dated_roles)-1):
        if dated_roles[i][1] > dated_roles[i+1][0]:
            hp += 1; flags.append("overlapping employment dates"); break

    if hp >= 2: return 0.01, hp, flags
    if hp == 1: return 0.55, hp, flags
    return 1.0, hp, flags

# ─────────────────────────────────────────────────────────────
# Reasoning Builder (rich, structured, no hallucinations)
# ─────────────────────────────────────────────────────────────

def build_reasoning(cid, prof, feats):
    bits = []
    yoe = prof.get("years_of_experience", 0) or 0
    title = prof.get("current_title","")
    company = prof.get("current_company","")
    loc = prof.get("location","")

    bits.append(f"{title}{' at ' + company if company else ''}, {yoe:.1f} yrs, {loc}.")

    evs = [("embeddings", feats["embed_ev"]), ("vector-DB/search", feats["vdb_ev"]),
           ("Python", feats["py_ev"]), ("ranking-eval/NDCG", feats["eval_ev"])]
    found = [f"{label} ({ev[1]}, {ev[2]}mo)" for label, ev in evs if ev]
    if found: bits.append("Matched: " + "; ".join(found[:3]) + ".")

    beh = feats["beh"]
    if beh["days_inactive"] <= 30 and beh["resp_rate"] >= 0.5:
        bits.append(f"Highly active ({beh['days_inactive']}d ago, {beh['resp_rate']:.0%} response rate).")
    elif beh["days_inactive"] > 90:
        bits.append(f"Caution: {beh['days_inactive']}d inactive.")

    if beh["saved"] >= 5:
        bits.append(f"Saved by {beh['saved']} recruiters in 30d (market-validated).")

    if feats["flags"]:
        non_hp = [f for f in feats["flags"] if "CONSISTENCY" not in f]
        if non_hp: bits.append("Note: " + non_hp[0] + ".")

    return " ".join(bits)[:480]

# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    rows, texts = [], []
    n = 0

    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            cand = json.loads(line)
            ft = text_blob(cand)
            full_text = ft.lower()

            must_have, nice, evs     = compute_technical_fit(cand, full_text)
            role_traj, title_fit, traj, is_stuffer = compute_role_trajectory(cand, full_text)
            quality, edu_s, comp_s, cert_s = compute_quality_signals(cand, full_text)
            behav_mult, beh_detail  = compute_behavioral_market(cand)
            logistics, log_detail   = compute_logistics(cand)
            penalty, pen_flags       = compute_penalty(cand, full_text, is_stuffer)
            hp_pen, hp_flags_n, hp_flags = compute_honeypot(cand)

            rows.append({
                "candidate_id": cand["candidate_id"],
                "must_have": must_have, "nice": nice,
                "role_traj": role_traj, "quality": quality, "logistics": logistics,
                "penalty": penalty, "behav_mult": behav_mult, "hp_pen": hp_pen,
                "embed_ev": evs[0], "vdb_ev": evs[1], "py_ev": evs[2], "eval_ev": evs[3],
                "beh": beh_detail, "flags": pen_flags + hp_flags, "hp_flags": hp_flags_n,
                "prof": cand["profile"], "text": full_text,
            })
            texts.append(ft)
            n += 1

    t1 = time.time()
    print(f"[1/4] Extracted features for {n:,} candidates in {t1-t0:.1f}s", file=sys.stderr)

    # TF-IDF semantic similarity
    vec = TfidfVectorizer(max_features=35000, ngram_range=(1,2), stop_words="english", min_df=2)
    tfidf = vec.fit_transform(texts + [JD_TEXT.lower()])
    jd_vec = tfidf[-1]; cand_vecs = tfidf[:-1]
    sims = (cand_vecs @ jd_vec.T).toarray().ravel()
    order = sims.argsort()
    pct = [0.0] * len(sims)
    for ri, idx in enumerate(order): pct[idx] = ri / max(1, len(sims)-1)
    t2 = time.time()
    print(f"[2/4] TF-IDF similarity computed in {t2-t1:.1f}s", file=sys.stderr)

    # Score all candidates
    scored = []
    for i, r in enumerate(rows):
        ts = pct[i]
        # Weighted combination across 5 layers
        base = (0.22*ts + 0.28*r["must_have"] + 0.18*r["role_traj"] +
                0.14*r["quality"] + 0.12*r["logistics"] + 0.06*r["nice"])
        final = base * r["penalty"] * r["behav_mult"] * r["hp_pen"]
        scored.append((final, ts, r["candidate_id"], r))

    scored.sort(key=lambda x: (-x[0], x[2]))
    top = scored[:args.top_n]
    t3 = time.time()
    print(f"[3/4] Scored & ranked {n:,} candidates in {t3-t2:.1f}s", file=sys.stderr)

    hp_in_top = sum(1 for _,_,_,r in top if r["hp_flags"] >= 2)
    print(f"      Honeypots in top {args.top_n}: {hp_in_top} ({hp_in_top/args.top_n*100:.0f}%)", file=sys.stderr)

    # Tie-break: equal rounded scores → candidate_id ascending
    max_s = max(s for s,*_ in top); min_s = min(s for s,*_ in top)
    span  = max(max_s - min_s, 1e-9)

    rounded = []
    for raw_s, ts, cid, r in top:
        d = round(0.40 + 0.59*(raw_s-min_s)/span, 4)
        rounded.append([d, cid, r])

    i = 0
    while i < len(rounded):
        j = i
        while j+1 < len(rounded) and rounded[j+1][0] == rounded[i][0]: j += 1
        if j > i: rounded[i:j+1] = sorted(rounded[i:j+1], key=lambda x: x[1])
        i = j+1

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id","rank","score","reasoning"])
        for rank_i, (disp_s, cid, r) in enumerate(rounded, start=1):
            reasoning = build_reasoning(cid, r["prof"], r)
            w.writerow([cid, rank_i, f"{disp_s:.4f}", reasoning])

    t4 = time.time()
    print(f"[4/4] Written to {args.out} in {t4-t3:.1f}s", file=sys.stderr)
    print(f"TOTAL: {t4-t0:.1f}s  (budget: 300s)", file=sys.stderr)

if __name__ == "__main__":
    main()
