from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


@dataclass(frozen=True)
class LibrarySpec:
    distribution: str
    import_name: str
    responsibility: str
    required: bool = True


LIBRARIES: tuple[LibrarySpec, ...] = (
    LibrarySpec("pandas", "pandas", "tabular/causal resampling"),
    LibrarySpec("pandas-ta-classic", "pandas_ta_classic", "technical indicators"),
    LibrarySpec("scikit-learn", "sklearn", "KNN/scalers/classical ML"),
    LibrarySpec("river", "river", "forward online metrics/drift"),
    LibrarySpec("vectorbt", "vectorbt", "research backtesting"),
    LibrarySpec("SQLAlchemy", "sqlalchemy", "persistent prediction/outcome storage"),
    LibrarySpec("APScheduler", "apscheduler", "shadow scheduling"),
    LibrarySpec("pydantic", "pydantic", "validated domain contracts"),
    LibrarySpec("MarketProfile", "market_profile", "volume profile", required=False),
    LibrarySpec(
        "smartmoneyconcepts",
        "smartmoneyconcepts",
        "SMC structures/OB/FVG",
        required=False,
    ),
)


class LibraryRegistry:
    """Single source of truth for reused third-party capabilities."""

    @staticmethod
    def status() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for spec in LIBRARIES:
            installed = True
            dist_version: str | None = None
            error: str | None = None
            try:
                import_module(spec.import_name)
                try:
                    dist_version = version(spec.distribution)
                except PackageNotFoundError:
                    dist_version = "unknown"
            except Exception as exc:  # import diagnostics are part of the report
                installed = False
                error = f"{type(exc).__name__}: {exc}"

            rows.append(
                {
                    "distribution": spec.distribution,
                    "import_name": spec.import_name,
                    "responsibility": spec.responsibility,
                    "required": spec.required,
                    "installed": installed,
                    "version": dist_version,
                    "error": error,
                }
            )
        return rows

    @classmethod
    def assert_required(cls) -> None:
        missing = [
            row["distribution"]
            for row in cls.status()
            if row["required"] and not row["installed"]
        ]
        if missing:
            raise RuntimeError(f"Missing required GTG libraries: {', '.join(missing)}")
