"""
schema.py — 報告層的資料結構：影像模型輸出 → 結節「事實」（facts）

報告裡的每一句話都必須能追溯到這裡的某個欄位，LLM 只能改寫，不能新增。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

NODULE_TYPES = ("solid", "part_solid", "ground_glass")


@dataclass
class Nodule:
    nodule_id: int
    center_mm: tuple[float, float, float]  # 世界座標 (x, y, z)，與 LUNA16 coordX/Y/Z 相同
    diameter_mm: float  # 平均直徑（軸向長短徑平均，或分割等體積球直徑）
    diameter_source: str  # "detector_box" | "segmentation" | "manual"
    det_prob: float  # MONAI 偵測器信心分數
    malignancy_prob: float | None = None  # LUNA25 惡性度模型分數（非診斷）
    volume_mm3: float | None = None  # LIDC 分割體積（若有）
    nodule_type: str = "solid"  # 偵測器無法分辨型態，預設 solid 並在報告中標註
    type_source: str = "assumed"  # "assumed" | "model" | "manual"
    solid_component_mm: float | None = None  # part-solid 才需要
    lobe: str | None = None  # 目前管線未提供，保留欄位

    def __post_init__(self) -> None:
        if self.nodule_type not in NODULE_TYPES:
            raise ValueError(f"nodule_type 必須是 {NODULE_TYPES}，收到 {self.nodule_type!r}")


@dataclass
class ModelInfo:
    detector: str = "MONAI lung_nodule_ct_detection (RetinaNet), LUNA16 CPM 0.888"
    malignancy: str | None = "LUNA25 ResNet-18 5-fold ensemble, NLST semi-external AUC 0.876"
    segmentation: str | None = None


@dataclass
class Study:
    study_id: str
    nodules: list[Nodule] = field(default_factory=list)
    det_threshold: float = 0.5
    models: ModelInfo = field(default_factory=ModelInfo)
    baseline: bool = True  # 目前只支援基線篩檢（無前次影像比較）

    def to_dict(self) -> dict:
        return asdict(self)
