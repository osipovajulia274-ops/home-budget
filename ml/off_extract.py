"""Extract (name, tags) pairs from Open*Facts dumps with DuckDB, and map tags -> subcategory.

The tag->subcategory mapping (`map_tags`) is pure Python and unit-tested. The DuckDB part
(`extract_*`) needs the real dump files and `pip install duckdb`; it has NOT been run against
the live dumps yet (see PROGRESS.md) - schema is always printed first, never guessed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "data" / "off_category_map.yaml"

CYR_RE = re.compile(r"[А-Яа-яЁё]")


def load_map(path: Path = MAP_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def has_russian(name: Optional[str], min_letters: int = 3) -> bool:
    return bool(name) and len(CYR_RE.findall(name)) >= min_letters


def map_tags(tags: Iterable[str], rules: Sequence[dict]) -> Optional[str]:
    """First matching rule wins; a rule with `default: true` matches anything (use last)."""
    tags = [t.strip().lower() for t in tags if t and t.strip()]
    for rule in rules:
        if rule.get("default"):
            return rule["to"]
        patterns = [p.lower() for p in rule.get("tags", [])]
        for tag in tags:
            if any(tag == p or tag.startswith(p + "-") or tag.startswith(p) for p in patterns):
                return rule["to"]
    return None


# --------------------------------------------------------------------------- DuckDB part

def _duckdb():
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("duckdb is required for this step: pip install -r requirements-train.txt") from exc
    return duckdb


def describe(path: str) -> List[dict]:
    """Print and return the column list (name, type) of a parquet/csv dump."""
    duckdb = _duckdb()
    con = duckdb.connect()
    src = _source_sql(path)
    rows = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
    cols = [{"name": r[0], "type": r[1]} for r in rows]
    for c in cols:
        print(f"  {c['name']:40s} {c['type']}")
    return cols


def _source_sql(path: str) -> str:
    p = path.replace("'", "''")
    if p.endswith(".parquet"):
        return f"read_parquet('{p}')"
    return (f"read_csv('{p}', delim='\\t', quote='', header=true, all_varchar=true, "
            f"ignore_errors=true, sample_size=-1)")


_CYR3 = "[а-яА-ЯёЁ].*[а-яА-ЯёЁ].*[а-яА-ЯёЁ]"


def _name_select(col_type: str) -> str:
    """SQL selecting (name, lang) from the real column type.

    D1 stores product_name as LIST(STRUCT(lang, text)) (verified in raw_schemas.json): prefer the entry
    tagged lang='ru', otherwise the first entry with Cyrillic text. D3-D5 are plain CSV strings.
    """
    if "STRUCT" in col_type.upper():
        ru = f"list_filter(product_name, x -> x.lang = 'ru' AND regexp_matches(x.\"text\", '{_CYR3}'))"
        anyc = f"list_filter(product_name, x -> regexp_matches(x.\"text\", '{_CYR3}'))"
        return f"COALESCE({ru}[1], {anyc}[1]) AS pick"
    return "{'lang': CAST(NULL AS VARCHAR), 'text': product_name} AS pick"


def _list_or_text(col: str, col_type: str) -> str:
    return f"array_to_string({col}, ',')" if col_type.endswith("[]") or "LIST" in col_type.upper() else col


def extract_names(path: str, kind: str, out_csv: str, limit: Optional[int] = None) -> int:
    """Write CSV (code, name, subcategory, tags, lang, countries) of Russian-named products.

    `lang` and `countries` let step 02 drop Bulgarian / Ukrainian names (see ml/name_clean.py); they are
    empty when the dump has no such column. `kind` selects the section of off_category_map.yaml.
    """
    duckdb = _duckdb()
    con = duckdb.connect()
    cols = {c["name"]: c["type"] for c in describe(path)}
    for required in ("product_name", "categories_tags"):
        if required not in cols:
            raise SystemExit(f"column {required!r} not found - fix off_extract.py to match the printed schema")
    tags_expr = _list_or_text("categories_tags", cols["categories_tags"])
    code_expr = "code" if "code" in cols else "NULL"
    countries_expr = _list_or_text("countries_tags", cols["countries_tags"]) if "countries_tags" in cols else "NULL"
    sql = f"""
        SELECT code, pick."text" AS name, pick.lang AS lang, tags, countries FROM (
            SELECT {code_expr} AS code, {_name_select(cols['product_name'])},
                   {tags_expr} AS tags, {countries_expr} AS countries
            FROM {_source_sql(path)}
            WHERE {tags_expr} IS NOT NULL AND {tags_expr} <> ''
        ) WHERE pick."text" IS NOT NULL
    """
    try:
        rows = con.execute(sql).fetchall()
    except Exception as exc:  # noqa: BLE001 - SQL for lang/countries is new and untested on live dumps
        print(f"[off_extract] extended query failed ({exc}); falling back to name+tags only")
        sql = f"""
            SELECT {code_expr} AS code, pick."text" AS name, CAST(NULL AS VARCHAR) AS lang, tags,
                   CAST(NULL AS VARCHAR) AS countries
            FROM (SELECT {code_expr}, {_name_select(cols['product_name'])}, {tags_expr} AS tags
                  FROM {_source_sql(path)} WHERE {tags_expr} IS NOT NULL AND {tags_expr} <> '')
            WHERE pick."text" IS NOT NULL
        """
        rows = con.execute(sql).fetchall()
    rules = load_map()[kind]
    out = []
    for code, name, lang, tags, countries in rows:
        if not has_russian(name):
            continue
        sub = map_tags(str(tags).split(","), rules)
        if sub:
            out.append((code, re.sub(r"\s+", " ", name).strip(), sub, str(tags), lang or "", countries or ""))
        if limit and len(out) >= limit:
            break
    import csv
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "subcategory", "tags", "lang", "countries"])
        w.writerows(out)
    print(f"[off_extract] {kind}: {len(out)} Russian-named rows -> {out_csv}")
    return len(out)


def write_schema_report(paths: dict, out_json: str) -> None:
    report = {}
    for kind, p in paths.items():
        if Path(p).exists():
            print(f"== {kind}: {p}")
            report[kind] = describe(p)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
