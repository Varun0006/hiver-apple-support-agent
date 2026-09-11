"""
End-to-End Evaluation Runner.

Phases:
  1. Baseline intent classifiers (Majority Class + TF-IDF LogReg) vs Golden Set
  2. Agent intent classifier vs Golden Set
  3. Agent escalation decisions vs expected_action in Golden Set
  4. LLM-as-Judge scoring on agent-generated replies
  5. Save failure cases for analysis
  6. Print consolidated benchmark report
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate_escalation import compute_escalation_metrics
from evaluation.evaluate_intent import compute_intent_metrics
from evaluation.evaluate_replies import aggregate_reply_scores
from evaluation.llm_judge import LLMJudge
from src.agent import SupportAgent
from src.intent.baseline import MajorityClassBaseline, TfIdfLogisticRegressionBaseline
from src.intent.intents import INTENT_TAXONOMY


# ------------------------------------------------------------------ #
# Helper: keyword intent matcher (used for weak training labels)
# ------------------------------------------------------------------ #
import re

def _match_intent(text: str) -> str:
    text_lower = str(text).lower()
    scores = {}
    for intent, meta in INTENT_TAXONOMY.items():
        s = sum(1 for kw in meta["keywords"] if re.search(r"\b" + re.escape(kw) + r"\b", text_lower))
        if s > 0:
            scores[intent] = s
    return max(scores, key=scores.get) if scores else "other_general"


# ------------------------------------------------------------------ #
# Phase 1 & 2: Baseline evaluations
# ------------------------------------------------------------------ #
def run_baseline_evaluation(X_test, y_test, X_train, y_train) -> dict:
    print("\n[Phase 1] Training baselines...")

    majority_model = MajorityClassBaseline()
    majority_model.fit(X_train, y_train)
    y_pred_maj = majority_model.predict(X_test)
    metrics_maj = compute_intent_metrics(y_test, y_pred_maj)

    tfidf_model = TfIdfLogisticRegressionBaseline(max_features=5000)
    tfidf_model.fit(X_train, y_train)
    y_pred_tfidf = tfidf_model.predict(X_test)
    metrics_tfidf = compute_intent_metrics(y_test, y_pred_tfidf)

    return {
        "majority": {"predictions": y_pred_maj, "metrics": metrics_maj},
        "tfidf": {"predictions": y_pred_tfidf, "metrics": metrics_tfidf},
    }


# ------------------------------------------------------------------ #
# Phase 3: Agent evaluation on Golden Set
# ------------------------------------------------------------------ #
def run_agent_evaluation(df_golden: pd.DataFrame, agent: SupportAgent) -> pd.DataFrame:
    print("\n[Phase 3] Running agent on 200 Golden Set examples...")
    results = []
    for i, row in df_golden.iterrows():
        msg = str(row["message"])
        result = agent.run(msg)
        results.append({
            "conversation_id": row["conversation_id"],
            "message": msg,
            "true_intent": row["intent"],
            "pred_intent": result["intent"],
            "confidence": result["confidence"],
            "true_action": row["expected_action"],
            "pred_action": result["escalation"]["decision"],
            "escalation_reason": result["escalation"]["reason"],
            "reply": result["reply"],
            "reply_grounded": result["reply_grounded"],
            "historical_grounded": result.get("historical_grounded", result["reply_grounded"]),
            "answer_supported": result.get("answer_supported", False),
            "support_reason": result.get("support_reason", ""),
            "grounding_reason": result.get("grounding_reason", ""),
            "generation_model": result.get("generation_model", ""),
            "evidence": json.dumps([
                {"query": e.get("initial_query", "")[:100], "score": e.get("similarity_score", 0)}
                for e in result["evidence"]
            ]),
            "evidence_list": result["evidence"],
        })
        if (i + 1) % 25 == 0:
            print(f"  Processed {i+1}/200...")

    return pd.DataFrame(results)


# ------------------------------------------------------------------ #
# Phase 4: LLM-as-Judge scoring
# ------------------------------------------------------------------ #
def run_judge_scoring(df_results: pd.DataFrame, judge: LLMJudge) -> pd.DataFrame:
    print("\n[Phase 4] Scoring replies with LLM Judge...")
    scored_rows = []
    for i, row in df_results.iterrows():
        evidence = json.loads(row.get("evidence", "[]"))
        scores = judge.evaluate_reply(
            message=row["message"],
            evidence=evidence,
            reply=row["reply"],
        )
        scored_rows.append({
            **row.to_dict(),
            "judge_correctness": scores["correctness"],
            "judge_groundedness": scores["groundedness"],
            "judge_helpfulness": scores["helpfulness"],
            "judge_tone": scores["tone"],
            "judge_hallucination": scores["hallucination"],
            "judge_reasoning": scores.get("reasoning", ""),
        })
        if (i + 1) % 25 == 0:
            print(f"  Judged {i+1}/{len(df_results)}...")

    return pd.DataFrame(scored_rows)


# ------------------------------------------------------------------ #
# Phase 5: Failure analysis
# ------------------------------------------------------------------ #
def save_failure_analysis(df_judged: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intent classification failures
    intent_failures = df_judged[df_judged["true_intent"] != df_judged["pred_intent"]].copy()
    intent_failures.to_csv(output_dir / "intent_failures.csv", index=False)

    # Escalation failures (wrong decision)
    esc_failures = df_judged[df_judged["true_action"] != df_judged["pred_action"]].copy()
    esc_failures.to_csv(output_dir / "escalation_failures.csv", index=False)

    # Low quality replies (average judge score < 3.0)
    score_cols = ["judge_correctness", "judge_groundedness", "judge_helpfulness",
                  "judge_tone", "judge_hallucination"]
    df_judged["avg_judge_score"] = df_judged[score_cols].mean(axis=1)
    low_quality = df_judged[df_judged["avg_judge_score"] < 3.0].sort_values("avg_judge_score")
    low_quality.to_csv(output_dir / "low_quality_replies.csv", index=False)

    # All failures combined (any error)
    all_failures = df_judged[
        (df_judged["true_intent"] != df_judged["pred_intent"]) |
        (df_judged["true_action"] != df_judged["pred_action"]) |
        (df_judged["avg_judge_score"] < 3.0)
    ].copy()
    all_failures.to_csv(output_dir / "failures.csv", index=False)

    print(f"\n  Intent classification errors: {len(intent_failures)}")
    print(f"  Escalation decision errors:   {len(esc_failures)}")
    print(f"  Low quality replies (<3.0):   {len(low_quality)}")
    print(f"  Total failures saved:         {len(all_failures)}")

    return intent_failures, esc_failures, low_quality


# ------------------------------------------------------------------ #
# Main runner
# ------------------------------------------------------------------ #
def main(
    processed_csv: str = "data/processed/apple_conversations.csv",
    golden_csv: str = "data/golden/golden_set.csv",
    output_dir: str = "experiments",
):
    start = time.time()
    exp_dir = Path(output_dir)
    exp_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  APPLE SUPPORT AGENT — FULL EVALUATION HARNESS")
    print("=" * 60)

    # ---- Load data ----
    golden_path = Path(golden_csv)
    if not golden_path.exists():
        print(f"Golden set not found: {golden_path}. Run scripts/create_golden_set.py first.")
        return

    df_golden = pd.read_csv(golden_path)
    X_test = df_golden["message"].astype(str).tolist()
    y_test = df_golden["intent"].astype(str).tolist()
    y_true_action = df_golden["expected_action"].astype(str).tolist()

    print(f"Loaded Golden Set: {len(df_golden)} examples across {df_golden['intent'].nunique()} intents.")

    # ---- Training data ----
    df_train = pd.read_csv(processed_csv)
    df_train_sample = df_train.sample(n=min(10000, len(df_train)), random_state=42)
    X_train = df_train_sample["initial_query"].astype(str).tolist()
    y_train = [_match_intent(q) for q in X_train]

    # ---- Phase 1 & 2: Baselines ----
    baseline_results = run_baseline_evaluation(X_test, y_test, X_train, y_train)

    # ---- Phase 3: Agent pipeline ----
    agent = SupportAgent(confidence_threshold=0.60, retrieval_top_k=5)
    agent.warm_up()
    df_agent = run_agent_evaluation(df_golden, agent)

    # Save raw agent results
    df_agent_save = df_agent.drop(columns=["evidence_list"], errors="ignore")
    df_agent_save.to_csv(exp_dir / "agent_results.csv", index=False)

    # Compute agent intent metrics
    agent_intent_metrics = compute_intent_metrics(
        y_test, df_agent["pred_intent"].tolist()
    )
    # Compute escalation metrics
    agent_esc_metrics = compute_escalation_metrics(
        y_true_action, df_agent["pred_action"].tolist()
    )

    # ---- Phase 4: LLM Judge ----
    judge = LLMJudge()
    judge_records = df_agent.to_dict(orient="records")
    scored_records = []
    for i, rec in enumerate(judge_records):
        evidence_list = rec.get("evidence_list", [])
        scores = judge.evaluate_reply(
            message=rec["message"],
            evidence=evidence_list,
            reply=rec["reply"],
        )
        scored_records.append({
            **{k: v for k, v in rec.items() if k != "evidence_list"},
            "judge_correctness": scores["correctness"],
            "judge_groundedness": scores["groundedness"],
            "judge_helpfulness": scores["helpfulness"],
            "judge_tone": scores["tone"],
            "judge_hallucination": scores["hallucination"],
            "judge_reasoning": scores.get("reasoning", ""),
        })
        if (i + 1) % 25 == 0:
            print(f"  Judged {i+1}/200...")

    df_judged = pd.DataFrame(scored_records)
    judged_path = exp_dir / "judged_results.csv"
    df_judged.to_csv(judged_path, index=False)

    # Aggregate reply quality
    score_cols = ["judge_correctness", "judge_groundedness", "judge_helpfulness",
                  "judge_tone", "judge_hallucination"]
    avg_scores = df_judged[score_cols].mean().to_dict()

    # ---- Phase 5: Failure analysis ----
    failures_dir = Path("evaluation")
    save_failure_analysis(df_judged.copy(), failures_dir)

    elapsed = time.time() - start

    # ---- Print Report ----
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    print("\n--- Intent Classification ---")
    print(f"{'Model':<32} | {'Accuracy':>8} | {'Macro-F1':>8}")
    print("-" * 58)
    print(f"{'Baseline 1 (Majority Class)':<32} | {baseline_results['majority']['metrics']['accuracy']:>8.4f} | {baseline_results['majority']['metrics']['macro_f1']:>8.4f}")
    print(f"{'Baseline 2 (TF-IDF + LogReg)':<32} | {baseline_results['tfidf']['metrics']['accuracy']:>8.4f} | {baseline_results['tfidf']['metrics']['macro_f1']:>8.4f}")
    print(f"{'Agent (Keyword Classifier)':<32} | {agent_intent_metrics['accuracy']:>8.4f} | {agent_intent_metrics['macro_f1']:>8.4f}")

    print("\n--- Escalation Policy ---")
    print(f"  Precision: {agent_esc_metrics['precision']:.4f}")
    print(f"  Recall:    {agent_esc_metrics['recall']:.4f}")
    print(f"  F1:        {agent_esc_metrics['f1']:.4f}")

    print("\n--- Reply Quality (LLM Judge, avg / 5) ---")
    for dim, score in avg_scores.items():
        label = dim.replace("judge_", "").capitalize()
        print(f"  {label:<16}: {score:.2f}")

    print(f"\n  Total evaluation time: {elapsed:.1f}s")
    print("=" * 60)

    # Save full summary
    summary = {
        "baseline_majority": baseline_results["majority"]["metrics"],
        "baseline_tfidf": baseline_results["tfidf"]["metrics"],
        "agent_intent": agent_intent_metrics,
        "agent_escalation": agent_esc_metrics,
        "reply_quality_avg": {k.replace("judge_", ""): round(v, 3) for k, v in avg_scores.items()},
    }
    with open(exp_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to: {exp_dir / 'evaluation_summary.json'}")

    # Also save baseline results CSV
    pd.DataFrame([
        {"model": "Majority Class", **baseline_results["majority"]["metrics"]},
        {"model": "TF-IDF + LogReg", **baseline_results["tfidf"]["metrics"]},
        {"model": "Agent", **agent_intent_metrics},
    ]).to_csv(exp_dir / "baseline_results.csv", index=False)


if __name__ == "__main__":
    main()
