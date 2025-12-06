from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class SpiderPaths:
    """Locations for the SPIDER dataset artifacts we use."""

    root: Path
    data_dir: Path | None = None
    tables_json: Path | None = None
    database_dir: Path | None = None

    def __post_init__(self) -> None:
        base = self.data_dir or (self.root / "spider_dataset" / "spider_data")
        self.data_dir = base
        self.tables_json = self.tables_json or (base / "tables.json")
        self.database_dir = self.database_dir or (base / "database")


@dataclasses.dataclass
class RetrievalConfig:
    """Settings for table retrieval."""

    top_k: int = 5
    top_k_sparse: int = 20
    top_k_dense: int = 20
    hybrid_weight_sparse: float = 0.5
    hybrid_weight_dense: float = 0.5
    hybrid_strategy: str = "weighted"  # "weighted" or "rrf"
    rrf_k: int = 60
    dense_model_name: str = "BAAI/bge-base-en"


@dataclasses.dataclass
class PipelineConfig:
    project_root: Path
    spider_paths: SpiderPaths
    retrieval: RetrievalConfig

    @classmethod
    def from_root(cls, root: str | Path) -> "PipelineConfig":
        root_path = Path(root).resolve()
        return cls(
            project_root=root_path,
            spider_paths=SpiderPaths(root_path),
            retrieval=RetrievalConfig(),
        )


DEFAULT_CONFIG = PipelineConfig.from_root(Path(__file__).resolve().parents[1])
