from __future__ import annotations

from typing import Any, Dict, List


_OP_SYMBOLS: Dict[str, str] = {
    "plus": "+",
    "minus": "-",
    "times": "*",
    "divide": "/",
}


def _token_to_text(tok: Dict[str, Any]) -> str:
    role = tok.get("role")
    detail = tok.get("detail")

    if role in {"main", "carry", "borrow"}:
        if detail is None:
            return "?"
        return str(detail)

    if role == "operator":
        if detail is None:
            return "?"
        return _OP_SYMBOLS.get(str(detail), str(detail))

    if role == "structure":
        # For now we only have result_bar as a structure in this codebase.
        if detail == "result_bar":
            return "|"
        return str(detail) if detail is not None else str(tok.get("label", "?"))

    return str(detail) if detail is not None else str(tok.get("label", "?"))


def format_parsed_equation_readable(equation_dict: Dict[str, Any]) -> str:
    """
    Best-effort human-readable rendering from the parser output dict.
    This is intentionally *not* LaTeX; it is only meant to show digits/operators/rows quickly.
    """
    kind = equation_dict.get("equation_kind", "unknown")
    rows: List[Dict[str, Any]] = equation_dict.get("rows", [])

    if not rows:
        return f"equation_kind: {kind}\n(no tokens)"

    lines: List[str] = [f"equation_kind: {kind}"]
    for row in rows:
        row_idx = row.get("row")
        tokens: List[Dict[str, Any]] = row.get("tokens", [])
        text_tokens = [_token_to_text(tok) for tok in tokens]
        prefix = f"row {row_idx}: " if row_idx is not None else ""
        lines.append(prefix + " ".join(t for t in text_tokens if t).strip())

    return "\n".join(lines).strip()

