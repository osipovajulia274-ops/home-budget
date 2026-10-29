import numpy as np
import pandas as pd
import pytest

from ml import forecast_data as fd
from ml.transactions import food_share, generate_history, load_config, split_users


@pytest.fixture(scope="module")
def data():
    return generate_history(n_users=40, seed=7)


def test_columns_and_dtypes(data):
    tx, users = data
    assert list(tx.columns) == ["user_id", "persona", "date", "subcategory", "category", "amount", "regular"]
    assert (tx["amount"] > 0).all()
    assert (tx["category"] == tx["subcategory"].str.split(".").str[0]).all()
    assert tx["date"].min() == pd.Timestamp("2024-10-01") and tx["date"].max() <= pd.Timestamp("2026-09-30")
    assert users["user_id"].is_unique


def test_deterministic_for_a_seed():
    a, _ = generate_history(n_users=5, seed=3)
    b, _ = generate_history(n_users=5, seed=3)
    c, _ = generate_history(n_users=5, seed=4)
    assert a.equals(b) and not a.equals(c)


def test_subcategories_exist_in_taxonomy(data):
    from ml.seed_generator import load_taxonomy

    tax = set(load_taxonomy())
    assert set(data[0]["subcategory"]) <= tax


def test_food_share_matches_rosstat_reference(data):
    # guide 7.3: groceries about one third of spending (regional 31-34%); the generator is calibrated to it
    share = food_share(data[0])
    assert 0.29 <= share <= 0.37, share


def test_regular_payments_are_monthly_and_on_a_fixed_day(data):
    tx = data[0]
    reg = tx[tx["subcategory"] == "subscriptions.streaming"]
    g = reg.groupby(["user_id", "amount"])      # one user can hold several subscriptions
    for uid, rows in g:
        days = rows["date"].dt.day
        assert days.nunique() <= 2          # only the month-end clamp can change the day
        assert len(rows) >= 20              # present in (almost) every one of the 24 months


def test_transfers_exist_but_are_not_spending(data):
    tx = data[0]
    assert (tx["subcategory"] == "other.transfers").any()
    users = pd.DataFrame({"user_id": tx["user_id"].unique(), "salary_days": "5"})
    series = fd.build_series(tx, users)
    spend_total = sum(s.daily.sum() for s in series.values())
    assert np.isclose(spend_total, tx.loc[tx["subcategory"] != "other.transfers", "amount"].sum())


def test_user_split_is_disjoint_and_complete():
    ids = [f"u{i}" for i in range(300)]
    s = split_users(ids, seed=1)
    assert sorted(s["train"] + s["val"] + s["test"]) == sorted(ids)
    assert len(set(s["train"]) & set(s["val"])) == len(set(s["train"]) & set(s["test"])) == len(set(s["val"]) & set(s["test"])) == 0
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (210, 45, 45)


def test_config_personas_are_complete():
    cfg = load_config()
    assert abs(sum(p["weight"] for p in cfg["personas"].values()) - 1.0) < 1e-9
    for name, p in cfg["personas"].items():
        assert set(p["shares"]) <= set(cfg["events"]), name
