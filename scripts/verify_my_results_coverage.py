#!/usr/bin/env python3
"""Verify abnormal-value coverage after normalize_my_results_explain."""
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
    normalize_my_results_extract,
    normalize_my_results_explain,
)


def covered_names(explain: dict) -> set[str]:
    names = set()
    for group in explain.get("resultGroups") or []:
        for name in group.get("testNames") or []:
            names.add(str(name).strip())
        for item in group.get("valueExplanations") or []:
            if isinstance(item, dict) and item.get("testName"):
                names.add(str(item["testName"]).strip())
    return names


def run_pdf(pdf_path: Path) -> None:
    meta = extract_text_from_pdf_with_meta(BytesIO(pdf_path.read_bytes()))
    text = meta.get("text") or ""
    print(f"\n{'=' * 72}\n{pdf_path.name} ({len(text)} chars)\n{'=' * 72}")

    extract_raw = ask_ai(
        f"{MY_RESULTS_ASSISTANT_SYSTEM}\n\n{MY_RESULTS_EXTRACT_PROMPT}",
        f"Document text:\n\n{text}",
    )
    extract = normalize_my_results_extract(extract_raw)
    abnormal = my_results_abnormal_test_names(extract)
    print(f"Abnormal values in extract: {len(abnormal)}")
    for name in abnormal:
        row = next(r for r in extract["results"] if r["name"] == name)
        print(f"  - {name}: {row.get('value')} {row.get('unit')} ({row.get('status')})")

    if not abnormal:
        explain = normalize_my_results_explain(
            {
                "explanation": "Summary.",
                "resultGroups": [],
                "questions": [{"text": "Follow-up?", "relatedCategory": "", "relatedTests": []}],
            },
            extract=extract,
        )
        print(f"Coverage check: no-op (0 abnormal). resultGroups={len(explain.get('resultGroups') or [])}")
        return

    payload = build_my_results_explain_payload(extract, patient_name="Test", known_conditions=[])
    explain_raw = ask_ai(
        f"{MY_RESULTS_ASSISTANT_SYSTEM}\n\n{MY_RESULTS_EXPLAIN_PROMPT}",
        payload,
    )
    explain = normalize_my_results_explain(explain_raw, extract=extract)
    covered = covered_names(explain)
    missing = [n for n in abnormal if n not in covered]
    fallback_groups = [
        g for g in explain.get("resultGroups") or []
        if g.get("category") == "Other flagged values"
    ]

    print(f"\nAfter normalize — covered: {len(covered)}/{len(abnormal)}")
    if fallback_groups:
        print(f"Fallback 'Other flagged values' group present with: {fallback_groups[0].get('testNames')}")
    if missing:
        print(f"STILL MISSING: {missing}")
    else:
        print("All abnormal values covered:")
        for name in abnormal:
            print(f"  ✓ {name}")


def main() -> None:
    samples = [
        Path("/Users/carolpradocasanova/Desktop/sample1_blood_test.pdf"),
        Path("/Users/carolpradocasanova/Desktop/sample2_clinic_letter.pdf"),
        Path("/Users/carolpradocasanova/Desktop/sample3_scan_report.pdf"),
    ]
    for path in sys.argv[1:] or []:
        run_pdf(Path(path))
    if len(sys.argv) <= 1:
        for sample in samples:
            if sample.exists():
                run_pdf(sample)


if __name__ == "__main__":
    main()
