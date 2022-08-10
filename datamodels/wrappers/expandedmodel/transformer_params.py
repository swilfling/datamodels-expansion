from dataclasses import dataclass
from .storage import JSONInterface
from dataclasses import field
import json


@dataclass
class TransformerParams(JSONInterface):
    type: str = ""
    params: dict = field(default_factory=dict)

    @staticmethod
    def store_parameters_list(parameters_list, path_full):
        with open(path_full, "w") as f:
            parameters = ",".join(params.to_json() for params in parameters_list)
            f.write("[" + parameters + "]")

    @classmethod
    def load_parameters_list(cls, path_full):
        with open(path_full, "r") as file:
            list_dicts = json.load(file)
            sim_param_list = [cls.from_json(dict) for dict in list_dicts]
            return sim_param_list

    @staticmethod
    def get_params_of_type(list_params, type: str):
        return [params for params in list_params if params.type == type]