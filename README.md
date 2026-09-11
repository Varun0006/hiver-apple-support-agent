# Hiver Apple Support Agent

![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9?logo=uv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9%2B-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.0%2B-150458?logo=pandas&logoColor=white)


**v0.1.0** — an evidence-aware support agent built on top of the AppleSupport conversations in the Customer Support on Twitter dataset. Give it a new customer message and it will classify what the person actually needs, pull up similar conversations Apple has handled before, draft a reply grounded in that history, and decide whether it's safe to send that reply on its own or whether a human needs to look at it first.

**Built with:** [uv](https://github.com/astral-sh/uv) · [scikit-learn](https://scikit-learn.org/) for TF-IDF retrieval and baselines · [pandas](https://pandas.pydata.org/) for conversation reconstruction · [google-genai](https://pypi.org/project/google-genai/) for optional LLM-backed classification, generation, and judging · [python-dotenv](https://pypi.org/project/python-dotenv/) for config

```text
customer message
  -> intent classification
  -> TF-IDF historical retrieval
  -> evidence-aware reply generation
  -> historical grounding + answer-support checks
  -> configurable escalation policy
  -> AUTO_HANDLE or ESCALATE, with a reason
```

## Problem Framing

### What "good" actually means here

I didn't want to just build something that guesses a label and calls it done. For AppleSupport specifically, a good system needs to do four things well, consistently:

1. Make sense of short, informal, misspelled, sometimes emotional Twitter messages and figure out the real issue underneath.
2. Use how Apple has actually responded in the past as real evidence, not just background flavor.
3. Give genuinely useful, concise help on the routine stuff — troubleshooting, App Store problems, subscriptions, basic how-to questions.
4. Know when *not* to act — anything account-specific or financial gets handed to a person, with a clear reason why.

The agent never pretends to have looked at someone's account, confirmed a charge, restored access, approved a refund, or verified an identity. And it doesn't make up policies, prices, guarantees, timelines, or links — if it can't back something up with real historical evidence, it says so instead of guessing.

### What I deliberately didn't build

This is not a production support tool, and I kept it that way on purpose. It doesn't send anything on Twitter, touch Apple's actual systems, recover accounts, issue refunds, verify identity, manage subscriptions, or make device changes. It also doesn't hold onto conversation state across messages — each run is a self-contained proposal: here's the intent, here's the evidence, here's the reply, here's the routing decision.

That scope felt right for this exercise. The interesting question isn't "can this send a tweet" — it's "can a messy, real-world dataset support a system whose decisions you can actually defend." So I put the effort into reproducible data prep, retrieval that you can inspect, honest evaluation, and a real look at where it breaks, rather than wiring it up to a live platform.

## Data and Method

I reconstructed the AppleSupport threads from the raw TWCS export into `data/processed/apple_conversations.csv` — full conversations, not isolated tweets, so a reply carries the customer's context along with Apple's actual response. The retrieval index ends up with 80,759 conversations.

For retrieval, I went with TF-IDF (word unigrams/bigrams) plus cosine similarity — nothing exotic. It's fast, fully inspectable, deterministic, and doesn't need a vector database to reproduce. For a take-home baseline, that felt like the right level of complexity, not the ceiling of what's possible.

Worth being upfront about the dataset's limits, because they shape a lot of downstream decisions:

- Tweets are short, informal, sometimes multilingual.
- A lot of "support replies" are just DM handoffs, not real resolutions.
- Old links in historical replies may be dead or unhelpful.
- Similar wording doesn't always mean the same underlying problem.
- Some threads are just incomplete.

That's exactly why the escalation policy leans conservative — the data itself doesn't always give you a clean, trustworthy answer to hand a customer.

## Installation

Requires Python 3.12 or newer. Run these commands from the repository root.

### Option A: uv (recommended)

Install `uv` on Windows if it is not already installed(powershell admerstrative mode):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell if `uv` is not recognized, then install the locked project dependencies:

```powershell
uv sync
```

Run the project through the managed environment with `uv run`:

```powershell
uv run python main.py --message "How do I take a screenshot on my iPhone?"
```

### Option B: pip and venv

Create and activate a local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell blocks activation, allow it for the current terminal session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

After activation, use `python` commands as shown below. To leave the environment later:

```powershell
deactivate
```

### Configuration and data
The submitted `archive.zip` contains the dataset files used by this project:
`twcs/twcs.csv` (the full Customer Support on Twitter export) and `sample.csv`
(a small sample for quick checks). Extract the archive from the repository root
so that the full dataset is available at if the archive does not work for you  can download it from [kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) :

```text
twcs/twcs.csv
```

On Windows PowerShell:

```powershell
Expand-Archive -Path archive.zip -DestinationPath . -Force
```

Build the processed conversation files before running the agent against the full
dataset:

```powershell
python scripts/prepare_data.py --input twcs/twcs.csv --output data/processed/apple_conversations.csv
```

The command creates both `data/processed/apple_conversations.csv` and
`data/processed/apple_conversations.json`. The generated processed files are
not required when using only the sample dataset.

For Gemini-backed classification, generation, and judging, copy `.env.example` to
`.env` and set `GEMINI_API_KEY`. The key is optional: without it, the project uses
deterministic keyword, template, and heuristic fallbacks. Those fallback results
must not be described as LLM-quality evidence.

## Usage

After installation and data preparation, run commands from the repository root.
The default CLI loads the processed conversation data generated above. If you
have not prepared the full dataset yet, run the preparation command in the
Configuration and data section first.

### Answer one customer message

```powershell
python main.py --message "How do I take a screenshot on my iPhone?"
```

The CLI prints the customer message, predicted intent and confidence, retrieval
evidence, decision and reason, drafted reply, historical-grounding status, and
answer-support status.

### Interactive mode

Process multiple messages in one run. Type `exit` or `quit` to stop:

```powershell
python main.py --interactive
```

### Smoke test

Run the built-in set of representative AppleSupport cases:

```powershell
python main.py
```

### Retrieval size

The CLI retrieves three historical cases by default. Use `--top-k` to change the
number of evidence cases shown and supplied to reply generation:

```powershell
python main.py --top-k 1 --message "I was charged twice for an Apple purchase."
```

`--top-k` must be at least `1`. Larger values provide more context but may also
include weaker or repetitive historical matches.

## Running the Evaluation

```powershell
python scripts/validate_golden_set.py --require-reviewed
python scripts/run_evaluation.py
python evaluation/judge_human_agreement.py
```

This produces baseline comparisons, agent metrics, judged replies, failure-analysis CSVs, and human-vs-judge agreement results under `experiments/` and `evaluation/`. For a completely fresh data prep run:

```powershell
python scripts/prepare_data.py --input twcs/twcs.csv --output data/processed/apple_conversations.csv
python scripts/create_golden_set.py
python scripts/validate_golden_set.py --require-reviewed
```

(Always review generated rows by hand before trusting them as ground truth.)

## Results vs. Baselines

Measured on the reviewed 200-example golden set, against `experiments/evaluation_summary.json`:

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority baseline | 0.135 | 0.0216 | 0.0321 |
| TF-IDF + Logistic Regression | 0.465 | 0.4265 | 0.4738 |
| Agent | 0.700 | 0.6781 | 0.7174 |

So the agent clears both a trivial baseline (majority class) and a reasonable, conventional one (TF-IDF + logistic regression) by a wide margin. Macro-F1 matters here as much as accuracy, since the classes aren't equally easy and accuracy alone can quietly hide poor performance on the smaller categories.

**Escalation performance:**

- Precision: `0.513` · Recall: `1.000` · F1: `0.6781`
- True positives: `99` · False positives: `94` · False negatives: `0` · True negatives: `7`

Zero missed escalations in this set — but the policy also sends nearly half its "safe" traffic to a human unnecessarily. That's a defensible trade-off for a prototype where safety comes first, but it's not an efficient end state.

**Reply quality (LLM judge, 1–5 scale):**

| Dimension | Score |
|---|---:|
| Correctness | 3.00 |
| Groundedness | 3.00 |
| Helpfulness | 2.82 |
| Tone | 3.80 |
| Hallucination | 5.00 |

**Human vs. judge agreement (50 rated replies):**

| Dimension | Agreement | MAE |
|---|---:|---:|
| Correctness | 100% | 0.44 |
| Groundedness | 100% | 1.00 |
| Helpfulness | 38% | 1.26 |
| Tone | 100% | 0.26 |
| Hallucination | 100% | 0.00 |

A few dimensions barely vary in this sample, which pushes their Pearson/Spearman correlation to zero — that's a quirk of low variance, not evidence the judge is broken. Percentage agreement and MAE tell the real story there. The one number that *does* worry me is helpfulness: only 38% agreement between the human rater and the judge. That's the clearest signal that "safe and polite" and "actually useful" aren't the same thing yet.

## Failure Analysis — Top 5

Pulled from `evaluation/intent_failures.csv`, `evaluation/escalation_failures.csv`, and `evaluation/failures.csv` (60 intent failures, 94 escalation failures, 130 combined).

**1. It over-escalates routine device complaints.**
Example: *"In other news, my iPhone battery sucks."* Reviewed label was `device_hardware_issue`, expected action `AUTO_HANDLE` — but the agent escalated because its own reply wasn't judged sufficiently supported, even though retrieval returned several closely matching battery-complaint cases.
*Hypothesis:* the policy treats imperfect grounding as a hard stop even when retrieval evidence is strong. Worth testing whether battery, charging, and ordinary App Store cases can be safely auto-handled with evidence-aware replies rather than reflexively routed to a human.

**2. How-to questions get mistaken for hardware problems.**
Example: *"How do I permanently turn off that 'live' thing on my iPhone camera?"* Reviewed as `device_how_to`, but predicted `device_hardware_issue` at 0.63 confidence, with a charging-related reply — despite retrieval finding a near-perfect match (similarity 1.000) on Live Photo questions.
*Hypothesis:* broad hardware vocabulary is crowding out a distinct camera-feature signal. Intent-aware reranking plus more reviewed how-to examples should tighten this boundary.

**3. Update regressions get confused with generic hardware issues.**
Example: *"Hey there! I updated to iOS 11.1.2 last night and today my device keeps restarting on its own."* Reviewed as `software_update_issue`, expected `AUTO_HANDLE` — but predicted as `device_hardware_issue` at 0.63 confidence, and the grounding check rejected the generated response, forcing an escalation.
*Hypothesis:* the classifier is counting overlapping words (restart, update, device symptoms) without weighting the causal phrase "after I updated." A temporal-update feature and better multi-intent prioritization would help separate these.

**4. Broad billing/service language creates real ambiguity.**
Example: *"Hi! I have a 2012 MacBook Pro without AppleCare. Is there anything that can be done, preferably free of charge?"* Reviewed as `payment_billing_issue`, predicted as `device_hardware_issue`. This one's a genuine taxonomy gray area, not a bad keyword match — repair cost, warranty questions, and disputed charges all currently share territory.
*Hypothesis:* the taxonomy needs clearer annotation guidance separating "how much will this repair cost," "I was charged for this," and "my device is broken."

**5. Replies can be safe without being useful.**
Example: *"On a perfectly preserved iPhone 6s battery with iOS 10.x lasting three days, iOS 11.3 drains battery in less than 20 hours."* — the reply asked about model and charger behavior instead of addressing the actual drain complaint. A similar pattern showed up on a screenshot-preview question. These are exactly the kind of cases dragging helpfulness down to 2.82/5 and human/judge agreement down to 38%.
*Hypothesis:* generation needs a "first useful action" rule — address the named symptom with the strongest available evidence *before* asking clarifying questions. Avoiding hallucination isn't the same as being helpful.

## What's Misleading About My Headline Number?

The 0.700 agent accuracy is real, but it's easy to over-read. It answers exactly one question — does the predicted label match the reviewed label — and nothing about whether the reply actually helps the customer, whether the retrieved evidence was relevant, or whether the auto-handle/escalate call was operationally sound.

A few reasons it can look stronger than the full system really is:

- **Small test set.** 200 examples is enough to be useful, not enough to be a production-scale claim.
- **Taxonomy dependence.** A different intent schema would move this number around, possibly a lot.
- **Some circularity.** Candidate sampling and the deterministic fallback classifier both lean on overlapping keyword vocabulary.
- **Narrow metric.** Accuracy says nothing about helpfulness, grounding quality, latency, cost, or how much human workload the system actually creates.

The escalation numbers make the gap concrete: recall is a clean 1.000, but precision is only 0.513, with 94 false-positive escalations. The system isn't missing anything it should catch — it's just erring hard toward caution, which means a lot of unnecessary human load in its current form.

The reply metrics tell a similar story from a different angle: hallucination scored a perfect 5.00, but helpfulness only averaged 2.82/5, with just 38% human/judge agreement on that dimension. A reply can be polite, careful, and never make anything up, and still fail to actually answer what the customer asked. So the honest summary is: this is a working, inspectable pipeline with genuinely solid intent classification — but reply usefulness and escalation calibration are still open problems, not solved ones.

## What I'd Do With One More Week

- **Calibrate escalation on held-out data** — tune confidence, similarity, and answer-support thresholds properly instead of relying on smoke-test intuition, aiming to cut those 94 false positives without opening up false negatives.
- **Make retrieval intent-aware** — rerank by whether the historical reply actually resolved the issue, not just lexical similarity, so a near-duplicate question with only a DM handoff doesn't outrank a slightly-less-similar case with a real answer.
- **Sharpen the taxonomy** — clearer guidance for billing vs. card/ATM, repair-cost vs. hardware, and update-regression vs. generic symptoms, based on the reviewed examples rather than one-off rules.
- **Fix reply usefulness** — add a first-useful-action check so responses address the named symptom before asking follow-up questions.
- **Strengthen grounding evaluation** — build a small hand-labeled claim/action support set and compare the rule-based fallback evaluator against structured Gemini grounding judgments.
- **Get a second human rater** — 38% helpfulness agreement is too weak to trust on its own; a written rubric and a second rater would go a long way.
- **Measure the operational stuff** — latency, token usage, API cost, retrieval time — so this can be judged as a system, not just a classifier.
- **Stress-test robustness** — paraphrases, typos, multilingual input, mixed-intent messages, vague or adversarial phrasing, to see if these results hold up outside the reviewed sample.

## Decision Log

The 10–15 non-obvious calls behind this project, and the reasoning:

1. **Why AppleSupport:** solid volume, and support conversations that follow patterns most people will recognize.
2. **Rebuilding threads first:** retrieval evidence needed the customer's context and Apple's actual reply, not a lone tweet.
3. **Keeping the taxonomy small:** a compact intent set keeps evaluation consistent instead of splintering into one label per topic.
4. **Expanding it only when needed:** device how-to, device storage, and card/ATM got added because the actual project behavior and reviewed examples clearly called for them — not speculatively.
5. **Draft first, then check by hand:** keyword-based candidate sampling is fine as a starting point, but nothing became ground truth without manual review.
6. **200 examples for the golden set:** comfortably inside the required 150–250 range, balancing coverage against review effort.
7. **TF-IDF for retrieval:** fast, inspectable, fully reproducible, and plenty for a take-home baseline — no need to reach for embeddings here.
8. **Richer evidence for generation:** the generator gets customer text, historical Apple response, similarity score, and a thread excerpt — not just an intent label.
9. **Separating grounding from support:** a safe, general how-to answer can be useful even if it never showed up verbatim in the Twitter evidence — those are two different questions.
10. **Conservative security policy:** account compromise, disputed/unrecognized transactions, duplicate billing — always escalated, no exceptions.
11. **Conservative device-failure handling:** unresponsive-after-update, black screen, stuck-on-logo cases get escalated whenever historical evidence isn't strong enough to justify confident automated troubleshooting.
12. **Honest fallbacks:** deterministic offline replies are kept, but they're clearly marked on whether they're actually historically grounded, and never dressed up as LLM-judge evidence.
13. **No invented promises:** never fabricate refunds, prices, guarantees, account actions, or URLs — use a real historical support action when one's available.
14. **Real baselines, not just a number:** compared against both a majority classifier and a TF-IDF logistic-regression classifier for a meaningful reference point.
15. **Checking against a human:** a separate 50-example human-rated sample, reported with agreement, MAE, and an explicit flag on where constant scores made correlation numbers misleading — rather than hiding behind one clean headline figure.

## Reproduction

```powershell
uv sync
python scripts/validate_golden_set.py --require-reviewed
python scripts/run_evaluation.py
python evaluation/judge_human_agreement.py
python main.py
```

Don't trust any accuracy claim above unless these commands have been rerun after a data or code change. For Gemini-backed runs, set `GEMINI_API_KEY` in `.env`; without one, the pipeline still runs on deterministic fallbacks — just don't describe those fallback results as LLM-quality evidence.
