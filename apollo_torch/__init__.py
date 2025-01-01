__version__ = "1.0"

# GaLore optimizer
from .adafactor import Adafactor as GaLoreAdafactor
from .adamw import AdamW as GaLoreAdamW
from .adamw8bit import AdamW8bit as GaLoreAdamW8bit

# apollo optimizer
from .apollo import AdamW as APOLLOAdamW
from .q_apollo import AdamW as QAPOLLOAdamW