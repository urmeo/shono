"""Validate declared model kinds and training audits for scored results."""

from collections.abc import Mapping

MODEL_KINDS = frozenset({"fine_tuned", "pretrained", "provider", "unknown"})


def validate_scored_model(model_kind: object, leakage_audit: object) -> None:
    if not isinstance(model_kind, str) or model_kind not in MODEL_KINDS:
        raise ValueError("model_kind must be fine_tuned, pretrained, provider or unknown")
    if model_kind == "unknown":
        raise ValueError("scored cells require a declared model_kind")
    if model_kind == "fine_tuned" and (
        not isinstance(leakage_audit, Mapping) or leakage_audit.get("status") != "reviewed"
    ):
        raise ValueError("scored fine-tuned cells require a reviewed training audit")
