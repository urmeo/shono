"""Explicit per-source permissions and unknown license facts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from shono.data.validation import json_object, real_number, text_value

_STATUSES = frozenset({"active", "eval-only", "excluded"})


@dataclass(frozen=True)
class DatasetLicense:
    """One registered source's license facts."""

    id: str
    name: str
    url: str
    license: str
    redistributable_audio: bool | None
    attribution_required: bool | None
    share_alike: bool | None
    access: str
    status: str
    hours: float | None
    domain: str
    version: str
    notes: str
    confidence: str

    def __post_init__(self) -> None:
        text_value(self.status, "license status")
        if self.status not in _STATUSES:
            raise ValueError(
                f"license {self.id!r} has status {self.status!r}; "
                f"expected one of {sorted(_STATUSES)}"
            )
        for key in ("id", "name", "license", "access", "domain", "version", "confidence"):
            text_value(getattr(self, key), f"license {key}")
        for key in ("url", "notes"):
            text_value(getattr(self, key), f"license {key}", empty=True)
        for key in ("redistributable_audio", "attribution_required", "share_alike"):
            value = getattr(self, key)
            if value is not None and type(value) is not bool:
                raise ValueError(f"license {key} must be a boolean or null (unknown)")
        if self.hours is not None and real_number(self.hours, "license hours") < 0:
            raise ValueError("license hours must be nonnegative")


class LicenseRegistry:
    """Immutable lookup of :class:`DatasetLicense` records, keyed by source id."""

    def __init__(self, licenses: Iterable[DatasetLicense], as_of: str = "") -> None:
        text_value(as_of, "registry as_of", empty=True)
        by_id: dict[str, DatasetLicense] = {}
        for lic in licenses:
            if not isinstance(lic, DatasetLicense):
                raise ValueError("registry entries must be DatasetLicense records")
            if lic.id in by_id:
                raise ValueError(f"duplicate license id {lic.id!r} in registry")
            by_id[lic.id] = lic
        self._by_id = by_id
        self.as_of = as_of

    @classmethod
    def load(cls, path: str | Path) -> LicenseRegistry:
        """Load the registry from a ``licenses.json`` document."""
        doc = json_object(Path(path).read_text(encoding="utf-8"), str(path))
        unknown = set(doc) - {"as_of", "note", "licenses"}
        if unknown or "licenses" not in doc:
            raise ValueError(f"{path}: invalid registry fields {sorted(unknown)}")
        records = doc["licenses"]
        if not isinstance(records, list) or not records:
            raise ValueError(f"{path}: licenses must be a non-empty array")
        fields = DatasetLicense.__dataclass_fields__
        licenses = []
        for i, rec in enumerate(records):
            if not isinstance(rec, dict) or set(rec) != set(fields):
                raise ValueError(f"{path}: license entry {i} needs exactly {sorted(fields)}")
            try:
                licenses.append(DatasetLicense(**rec))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: license entry {i}: {exc}") from exc
        if "note" in doc:
            text_value(doc["note"], "registry note", empty=True)
        return cls(licenses, as_of=doc.get("as_of", ""))

    def __contains__(self, source_id: object) -> bool:
        return source_id in self._by_id

    def __getitem__(self, source_id: str) -> DatasetLicense:
        return self._by_id[source_id]

    def __iter__(self):
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def require(self, source_id: str) -> DatasetLicense:
        """Return the license for ``source_id`` or raise with an actionable message."""
        try:
            return self._by_id[source_id]
        except KeyError:
            raise KeyError(
                f"source {source_id!r} is not in the license registry; "
                f"register it in data/licenses.json before referencing it "
                f"(known: {sorted(self._by_id)})"
            ) from None

    def sources(self, status: str | None = None) -> list[str]:
        """Registered source ids, optionally filtered to one ``status``."""
        return sorted(
            lic.id for lic in self._by_id.values() if status is None or lic.status == status
        )

    def render_table(self) -> str:
        """Render the license table as GitHub-flavoured Markdown, sorted by status then id."""
        header = (
            "| Source | Hours | Domain | License | Redistribute audio? | Status |\n"
            "|---|---|---|---|---|---|"
        )
        order = {"active": 0, "eval-only": 1, "excluded": 2}
        rows = []
        for lic in sorted(self._by_id.values(), key=lambda x: (order.get(x.status, 9), x.id)):
            hours = "unknown" if lic.hours is None else f"{lic.hours:g}"
            audio = (
                "unknown"
                if lic.redistributable_audio is None
                else "yes"
                if lic.redistributable_audio
                else "no"
            )
            link = f"[{lic.name}]({lic.url})" if lic.url else lic.name
            rows.append(
                f"| {link} | {hours} | {lic.domain} | {lic.license} | {audio} | {lic.status} |"
            )
        stamp = f"\n\n_Registry as of {self.as_of}._" if self.as_of else ""
        return f"{header}\n" + "\n".join(rows) + stamp
