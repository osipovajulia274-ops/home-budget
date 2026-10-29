"""Stage 2: generate the SYNTHETIC transaction history used for the forecaster.

  python -m ml.scripts.05_generate_transactions [--users 300] [--seed 42]
    -> ml/data/processed/transactions_synth.parquet   (user_id, persona, date, subcategory, category, amount, regular)
    -> ml/data/processed/users_synth.csv              (ground-truth persona parameters, for tests/inspection)
    -> ml/reports/transactions_synth_summary.json     (food share, spend by category, users per split)

Persona parameters live in data/personas.yaml. The data is synthetic and must be described as such in the
report; it is not a measurement of real households.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.tableio import write_table  # noqa: E402
from ml.transactions import food_share, generate_history, split_users  # noqa: E402

PROCESSED = ROOT / "ml" / "data" / "processed"
REPORTS = ROOT / "ml" / "reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(PROCESSED / "transactions_synth.parquet"))
    args = ap.parse_args()

    tx, users = generate_history(args.users, args.seed)
    path = write_table(tx, args.out)
    users.to_csv(PROCESSED / "users_synth.csv", index=False, encoding="utf-8")

    spend = tx[tx["subcategory"] != "other.transfers"]
    by_cat = (spend.groupby("category")["amount"].sum() / spend["amount"].sum()).round(4).sort_values(ascending=False)
    splits = split_users(users["user_id"].tolist(), args.seed)
    summary = {
        "synthetic": True,
        "users": int(users["user_id"].nunique()),
        "rows": int(len(tx)),
        "date_from": str(tx["date"].min().date()), "date_to": str(tx["date"].max().date()),
        "food_share_of_spending": round(food_share(tx), 4),
        "rosstat_reference_food_share": "0.31-0.34 (regional household budget surveys 2019-2023)",
        "share_by_category": by_cat.to_dict(),
        "users_per_persona": users["persona"].value_counts().to_dict(),
        "users_per_split": {k: len(v) for k, v in splits.items()},
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "transactions_synth_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[05] {len(tx)} synthetic transactions of {summary['users']} users -> {path}")
    print(f"[05] food share of spending: {summary['food_share_of_spending']:.1%} (target ~ one third)")


if __name__ == "__main__":
    main()
