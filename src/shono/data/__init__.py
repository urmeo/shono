"""Dataset layer: manifests (audio-free slice descriptions) and the license registry.

Audio and checkpoints never enter git — this package models the *metadata* that
does: what a slice contains, where its audio lives, and under which license.
"""

from shono.data.leakage import LeakageReport, Overlap, audit_leakage
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
]
