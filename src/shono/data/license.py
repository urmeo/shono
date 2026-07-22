"""Per-source license registry — the single authority for the dataset license floor.

Every dataset a manifest references must be registered here first. The registry
is loaded from ``data/licenses.json`` (facts live in data, code renders them —
never the reverse), and it answers three questions the rest of the pipeline
asks:

    * Is this source allowed into the mix, and at what tier (``status``)?
    * What obligations does its license carry (attribution, share-alike)?
    * May its raw audio be redistributed? (Shono never ships audio regardless;
      the flag lets the ship gate double-check the obligation is honoured.)

The license *table* in reports and docs is rendered from this registry, so the
numbers and the licenses can never drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_STATUSES = frozenset({"active", "eval-only", "excluded"})


@dataclass(frozen=True)
class DatasetLicense:
    """One registered source's license facts.

    ``status`` encodes the license floor directly: ``active`` sources may be
    used for training and evaluation, ``eval-only`` sources for evaluation but
    never training (e.g. a private eval set), ``excluded`` sources for neither
    until their blocker (e.g. an unconfirmed license) clears.
    """

    id: str
    name: str
    url: str
    license: str
    redistributable_audio: bool
    attribution_required: bool
    share_alike: bool
    access: str
    status: str
    hours: float | None
    domain: str
    version: str
    notes: str
    confidence: str

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(
                f"license {self.id!r} has status {self.status!r}; "
                f"expected one of {sorted(_STATUSES)}"
            )
        if not self.id:
            raise ValueError("a license record needs a non-empty id")


class LicenseRegistry:
    """Immutable lookup of :class:`DatasetLicense` records, keyed by source id."""

    def __init__(self, licenses: Iterable[DatasetLicense], as_of: str = "") -> None:
        by_id: dict[str, DatasetLicense] = {}
        for lic in licenses:
            if lic.id in by_id:
                raise ValueError(f"duplicate license id {lic.id!r} in registry")
            by_id[lic.id] = lic
        self._by_id = by_id
        self.as_of = as_of

    @classmethod
    def load(cls, path: str | Path) -> LicenseRegistry:
        """Load the registry from a ``licenses.json`` document."""
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        records = doc["licenses"]
        fields = DatasetLicense.__dataclass_fields__
        licenses = [
            DatasetLicense(**{k: v for k, v in rec.items() if k in fields}) for rec in records
        ]
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
            hours = "—" if lic.hours is None else f"{lic.hours:g}"
            audio = "yes" if lic.redistributable_audio else "no"
            link = f"[{lic.name}]({lic.url})" if lic.url else lic.name
            rows.append(
                f"| {link} | {hours} | {lic.domain} | {lic.license} | {audio} | {lic.status} |"
            )
        stamp = f"\n\n_Registry as of {self.as_of}._" if self.as_of else ""
        return f"{header}\n" + "\n".join(rows) + stamp
