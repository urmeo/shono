"""Synthetic permissions for mechanics tests; they confer no source-data rights."""

from shono.data import DatasetLicense, LicenseRegistry


def synthetic_registry(*sources: str, status: str = "active") -> LicenseRegistry:
    return LicenseRegistry(
        DatasetLicense(
            id=source,
            name="Synthetic fixture",
            url="",
            license="fixture-only",
            redistributable_audio=False,
            attribution_required=False,
            share_alike=False,
            access="unit test",
            status=status,
            hours=None,
            domain="synthetic",
            version="fixture",
            notes="Synthetic test metadata, no real-source permission.",
            confidence="test",
        )
        for source in sources
    )
