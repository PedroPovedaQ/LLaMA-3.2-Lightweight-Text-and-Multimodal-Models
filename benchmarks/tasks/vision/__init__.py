from .mmmu import MMMUTask
from .mmmu_pro import MMMUProStandardTask, MMMUProVisionTask
from .mathvista import MathVistaTask
from .chartqa import ChartQATask
from .ai2_diagram import AI2DiagramTask
from .docvqa import DocVQATask
from .vqav2 import VQAv2Task

VISION_TASKS = {
    "mmmu": MMMUTask,
    "mmmu_pro_standard": MMMUProStandardTask,
    "mmmu_pro_vision": MMMUProVisionTask,
    "mathvista": MathVistaTask,
    "chartqa": ChartQATask,
    "ai2_diagram": AI2DiagramTask,
    "docvqa": DocVQATask,
    "vqav2": VQAv2Task,
}
