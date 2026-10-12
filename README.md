# DiploMind

**Adaptive AI-powered International Relations literacy platform for multilingual Pakistani users.**

DiploMind personalizes both the **language** (English / Urdu / Roman Urdu) and the **explanation depth** of IR and current-affairs content for each reader, using a continuously updated knowledge base and a feedback loop grounded in actual comprehension.

> FAST-NUCES Lahore, Department of Computer Science. Advisor: Dr. Hajra Waheed.
> Team: Muhammad Ibrahim, Bisma Amir, Maryum Nauman.
> Status: 🚧 work in progress

---

## Why DiploMind?

IR content for Pakistani readers is mostly English-first and written at one fixed level:

- **News** moves fast and assumes background knowledge.
- **Think-tank / academic writing** is dense, English-only, and written for specialists.

Readers who prefer Urdu or Roman Urdu, or who need more (or less) context, are poorly served. Most personalization systems also lock a user profile at onboarding and tune one preference at a time. DiploMind instead:

1. Keeps its knowledge base **current**.
2. Personalizes **language and depth independently**.
3. Keeps **updating each reader's profile** as evidence builds up.

---

## Architecture

The system has four parts:

| Component | What it does |
|---|---|
| **Offline data pipeline** | IR data scraper → clean & preprocess → dataset builder. Outputs the knowledge base, fine-tuning pairs, and a diagnostic quiz bank. |
| **Cold start (per new user)** | Diagnostic quiz → clustering → stored user profile. |
| **Online serving (per request)** | User query → document retrieval → personalization router (LinUCB bandit) → LLM + LoRA adapters (ΔW = language + depth) → adapted response. |
| **Feedback + retrain loop** | Post-explanation quiz → reward calculator → reward log. *Fast loop:* bandit weight updates. *Slow loop:* batched P-GRPO retraining of per-cluster LoRA adapters. A reassignment monitor updates the user profile on drift. |

---

## Project Targets

- [ ] **Dataset foundation:** continuously source current-affairs and foundational IR content (Pakistani and international; English and Urdu) and build language- and depth-adapted training data for adapter fine-tuning.
- [ ] **Cold-start onboarding:** a diagnostic quiz that sets a reasonable initial personalization profile before interaction history exists.
- [ ] **Personalized delivery and adaptive routing:** adapt language and depth independently at serving time, refining preferences from ongoing quiz feedback instead of relying only on the initial profile.
- [ ] **Drift detection:** a reward-based mechanism that detects when a reader's preferences have diverged from their profile cluster and reassigns them.

**Open research question:** can language and depth adapters be composed independently in an axis-specific setting like this one?

---

## Work Done So Far

- [x] Literature review (contextual bandits, parameter-efficient adaptation, learner personalization)
- [x] System architecture finalized
- [x] Data scraping pipeline started
- [x] Initial knowledge base populated with current-affairs and foundational IR content from local and international sources
- [ ] Adapter training dataset (**next**)
- [ ] Retrieval layer (**next**)
- [ ] Diagnostic quiz bank and cold-start clustering
- [ ] LoRA adapters (language + depth)
- [ ] LinUCB personalization router
- [ ] Feedback loop (reward calculator, bandit updates, P-GRPO retraining)
- [ ] Reassignment monitor (drift detection)
- [ ] End-to-end evaluation and demo

---

## Tech Stack

Python, PyTorch, Hugging Face (Transformers, PEFT), FastAPI.

---

## Key References

1. Li et al., *A contextual-bandit approach to personalized news article recommendation*, WWW '10. (LinUCB)
2. Zhang et al., *Composing Parameter-Efficient Modules with Arithmetic Operations*, NeurIPS 2023.
3. Huang et al., *LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition*, arXiv 2023.
4. Lim et al., *Personalized Learning Path Planning through Goal-Driven Learner State Modeling*, WWW '26. (Pxplore)

---

## Team

| Name | Roll No. |
|---|---|
| Muhammad Ibrahim | 23L-0521 |
| Bisma Amir | 23L-0659 |
| Maryum Nauman | 23L-0696 |

Advisor: Dr. Hajra Waheed
