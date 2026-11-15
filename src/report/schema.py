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
    # --- 追蹤檢查（非基線）才有的欄位，來自 scripts/nlst_longitudinal_tracks.py 的 tracks.csv ---
    comparable: bool | None = None  # False = 沒有可用的前次影像對位，改用基線規則
    is_new: bool | None = None  # 前次影像沒有對應結節
    new_confident: bool | None = None  # 追蹤管線的 is_new_confident；False = 可能是漏配，不套用新生規則
    prior_nearest_mm: float | None = None  # 對位後到前次任一候選的最近距離
    prior_diameter_mm: float | None = None
    days_since_prior: float | None = None
    volume_change_pct: float | None = None
    vdt_days: float | None = None  # 體積倍增時間
    upstream_growth_flag: bool | None = None  # 追蹤管線自己算的 lungrads_growth，用來交叉核對

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
    baseline: bool = True  # True = 基線篩檢；False = 追蹤檢查，有前次影像可比較
    exam_index: int | None = None  # 0 = T0，1 = T1 ...

    def to_dict(self) -> dict:
        return asdict(self)
