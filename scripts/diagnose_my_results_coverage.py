#!/usr/bin/env python3
"""One-off diagnostic: compare Call 2 raw resultGroups vs normalized coverage."""
import json
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_helpers import (  # noqa: E402
    MY_RESULTS_ASSISTANT_SYSTEM,
    MY_RESULTS_EXPLAIN_PROMPT,
    MY_RESULTS_EXTRACT_PROMPT,
    ask_ai,
    build_my_results_explain_payload,
    extract_text_from_pdf_with_meta,
    my_results_abnormal_test_names,
    my_results_extract_test_names,
    normalize_my_results_extract,
    normalize_my_results_explain,
    resolve_my_results_test_name,
)


def _collect_raw_group_names(raw_groups) -> dict:
    """Map raw LLM name -> resolution outcome."""
    out = {"testNames": [], "valueExplanations": [], "relatedTests": []}
    if not isinstance(raw_groups, list):
        return out
    for group in raw_groups:
        if not isinstance(group, dict):
            continue
        for field in ("testNames",):
            for name in group.get(field) or []:
                out[field].append(str(name))
        for item in group.get("valueExplanations") or []:
            if isinstance(item, dict) and item.get("testName"):
                out["valueExplanations"].append(str(item["testName"]))
    return out


def _collect_normalized_covered(result_groups) -> set[str]:
    covered = set()
    for group in result_groups or []:
        for name in group.get("testNames") or []:
            covered.add(str(name).strip().lower())
        for item in group.get("valueExplanations") or []:
            if isinstance(item, dict) and item.get("testName"):
                covered.add(str(item["testName"]).strip().lower())
    return covered


def _diagnose_name_resolution(raw_name: str, extract_names: list[str]) -> dict:
    resolved = resolve_my_results_test_name(raw_name, extract_names)
    return {
        "raw": raw_name,
        "resolved": resolved,
        "outcome": "matched" if resolved else "DROPPED",
    }


def run(pdf_path: Path) -> None:
    pdf_bytes = pdf_path.read_bytes()
    meta = extract_text_from_pdf_with_meta(BytesIO(pdf_bytes))
    text = meta.get("text") or ""
    print(f"PDF: {pdf_path.name} ({len(text)} chars extracted)\n")

    extract_raw = ask_ai(
        f"{MY_RESULTS_ASSISTANT_SYSTEM}\n\n{MY_RESULTS_EXTRACT_PROMPT}",
        f"Document text:\n\n{text}",
    )

    extract = normalize_my_results_extract(extract_raw)
    abnormal = my_results_abnormal_test_names(extract)
    extract_names = my_results_extract_test_names(extract)

    print("=== EXTRACT: abnormal values ===")
    for row in extract.get("results") or []:
        if str(row.get("status", "")).lower() in ("high", "low", "abnormal"):
            print(
                f"  - {row.get('name')}: {row.get('value')} {row.get('unit')} "
                f"({row.get('status')}) ref {row.get('referenceRange')}"
            )
    print(f"\nTotal abnormal: {len(abnormal)}\n")

    payload = build_my_results_explain_payload(extract, patient_name="Test", known_conditions=[])
    explain_raw = ask_ai(
        f"{MY_RESULTS_ASSISTANT_SYSTEM}\n\n{MY_RESULTS_EXPLAIN_PROMPT}",
        payload,
    )

    raw_groups = explain_raw.get("resultGroups") or []
    raw_names = _collect_raw_group_names(raw_groups)
    all_raw_mentions = set()
    for names in raw_names.values():
        all_raw_mentions.update(names)

    print("=== CALL 2 RAW: names mentioned in resultGroups ===")
    potassium_in_raw = False
    for group in raw_groups:
        if not isinstance(group, dict):
            continue
        cat = group.get("category", "")
        tests = group.get("testNames") or []
        vex = [item.get("testName") for item in (group.get("valueExplanations") or []) if isinstance(item, dict)]
        print(f"  Group '{cat}': testNames={tests}, valueExplanations={vex}")
        if any("potassium" in str(t).lower() for t in tests + vex):
            potassium_in_raw = True

    print(f"\nPotassium mentioned in raw Call 2 resultGroups: {potassium_in_raw}")

    print("\n=== VALIDATION: name resolution for each raw mention ===")
    dropped = []
    matched = []
    for raw_name in sorted(all_raw_mentions):
        diag = _diagnose_name_resolution(raw_name, extract_names)
        if diag["outcome"] == "DROPPED":
            dropped.append(diag)
        else:
            matched.append(diag)
        print(f"  {diag['raw']!r} -> {diag['resolved']!r} [{diag['outcome']}]")

    explain = normalize_my_results_explain(explain_raw, extract=extract)
    covered = _collect_normalized_covered(explain.get("resultGroups"))

    print("\n=== AFTER normalize_my_results_explain: covered abnormal ===")
    uncovered = []
    for name in abnormal:
        key = name.lower()
        status = "COVERED" if key in covered else "MISSING"
        print(f"  {name}: {status}")
        if key not in covered:
            uncovered.append(name)

    print("\n=== DIAGNOSIS ===")
    if not potassium_in_raw:
        print("(a) GENERATION GAP: Call 2 never mentioned Potassium in any resultGroup.")
    else:
        print("(b) MATCHING BUG: Call 2 mentioned Potassium but it was dropped during validation.")
        for d in dropped:
            if "potassium" in d["raw"].lower():
                print(f"    Drop detail: {d}")

    if uncovered:
        print(f"\nUncovered abnormal values ({len(uncovered)}): {uncovered}")
    else:
        print("\nAll abnormal values covered after normalization.")


if __name__ == "__main__":
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/carolpradocasanova/Desktop/sample1_blood_test.pdf")
    run(pdf)
