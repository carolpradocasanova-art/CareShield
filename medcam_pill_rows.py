"""MedCam verification pill-row HTML — Review before giving status cards."""

from __future__ import annotations

import html

MEDCAM_ROW_CHEVRON_SVG = (
    '<svg class="cs-medcam-pill-row-chevron" width="16" height="16" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true">'
    '<polyline points="6 9 12 15 18 9"/></svg>'
)


def build_medcam_reference_thumb_html(
    med_name: str | None,
    med_refs: list | None,
    *,
    find_reference,
) -> str:
    """Registered front-view thumbnail for an identified medication, or empty string."""
    name = str(med_name or "").strip()
    if not name or not med_refs:
        return ""
    ref = find_reference(name, med_refs)
    if not ref:
        return ""
    image_b64 = str(ref.get("image_b64") or "").strip()
    if not image_b64:
        return ""
    safe_name = html.escape(name)
    return (
        f'<img class="cs-medcam-pill-row-ref-thumb" '
        f'src="data:image/jpeg;base64,{image_b64}" '
        f'alt="{safe_name} reference photo" loading="lazy">'
    )


def build_medcam_row_html_from_display(row: dict, *, thumb_html: str = "") -> str:
    thumb_block = thumb_html if thumb_html else ""
    role = html.escape(str(row.get("role") or "neutral"))
    label = html.escape(str(row.get("label") or ""))
    verdict = html.escape(str(row.get("verdict") or ""))
    detail = html.escape(str(row.get("detail") or ""))
    parts = [
        f'<details class="cs-medcam-pill-row cs-medcam-pill-row--{role}">',
        '<summary class="cs-medcam-pill-row-summary">',
        '<div class="cs-medcam-pill-row-summary-inner">',
        '<div class="cs-medcam-pill-row-main">',
        f'<div class="cs-medcam-pill-row-label">{label}</div>',
        f'<div class="cs-medcam-pill-row-verdict">{verdict}</div>',
        '</div>',
    ]
    if thumb_block:
        parts.append(thumb_block)
    parts.extend([
        '</div>',
        MEDCAM_ROW_CHEVRON_SVG,
        '</summary>',
        f'<div class="cs-medcam-pill-row-detail">{detail}</div>',
        '</details>',
    ])
    return "".join(parts)
