"""Fail-closed promotion gate for a candidate model evaluation report."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def approve_candidate(candidate: dict, champion: dict, max_relative_regression: float = 0.0) -> tuple[bool, list[str]]:
    reasons = []
    candidate_rmse, champion_rmse = candidate.get("validation_rmse"), champion.get("validation_rmse")
    if not isinstance(candidate_rmse, (int, float)) or candidate_rmse < 0:
        reasons.append("candidate validation_rmse is missing or invalid")
    if not isinstance(champion_rmse, (int, float)) or champion_rmse < 0:
        reasons.append("champion validation_rmse is missing or invalid")
    if reasons:
        return False, reasons
    if candidate_rmse > champion_rmse * (1 + max_relative_regression):
        reasons.append(f"candidate RMSE {candidate_rmse:.4f} exceeds allowed limit {champion_rmse * (1 + max_relative_regression):.4f}")
    if candidate.get("schema_valid") is not True:
        reasons.append("candidate schema validation is missing or failed")
    if candidate.get("smoke_test_passed") is not True:
        reasons.append("candidate smoke-test evidence is missing or failed")
    return not reasons, reasons or ["candidate passed configured promotion checks"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--output", default="monitoring/reports/retraining_gate.json")
    parser.add_argument("--max-relative-regression", type=float, default=0.0)
    args = parser.parse_args()
    candidate = json.loads(Path(args.candidate).read_text())
    champion = json.loads(Path(args.champion).read_text())
    approved, reasons = approve_candidate(candidate, champion, args.max_relative_regression)
    result = {"approved": approved, "reasons": reasons, "candidate": args.candidate, "champion": args.champion}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0 if approved else 2


if __name__ == "__main__":
    raise SystemExit(main())
