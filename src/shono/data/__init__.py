"""Dataset metadata, format adapters and preflight checks."""

from shono.data.audio_paths import (
    validate_audio_paths,
    validate_audio_window,
    validate_manifest_audio,
)
from shono.data.build import (
    build_manifest,
    collapse_to_recordings,
    duration_from_audio,
    from_common_voice_tsv,
    from_fleurs_tsv,
    from_loop_jsonl,
    from_mucs_kaldi,
    from_slr53_tsv,
)
from shono.data.leakage import LeakageReport, Overlap, audit_leakage, require_no_leakage
from shono.data.license import DatasetLicense, LicenseRegistry
from shono.data.manifest import Manifest, Segment

__all__ = [
    "DatasetLicense",
    "LeakageReport",
    "LicenseRegistry",
    "Manifest",
    "Overlap",
    "Segment",
    "audit_leakage",
    "build_manifest",
    "collapse_to_recordings",
    "duration_from_audio",
    "from_common_voice_tsv",
    "from_fleurs_tsv",
    "from_loop_jsonl",
    "from_mucs_kaldi",
    "from_slr53_tsv",
    "require_no_leakage",
    "validate_audio_paths",
    "validate_audio_window",
    "validate_manifest_audio",
]
