from dataclasses import dataclass
from .storage import JSONInterface
from dataclasses import field

@dataclass
class TransformerParams(JSONInterface):
    type: str = ""
    params: dict = field(default_factory=dict)