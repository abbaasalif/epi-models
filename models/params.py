import json

class CampParams(object):
    def __init__(self, input_params: dict):
        for k, v in input_params.items():
            setattr(self, k, v)

    @classmethod
    def load_from_csv(cls):
        pass

    @classmethod
    def load_from_json(cls, file_path):
        with open(file_path) as file:
            input_params = json.load(file)
        return cls(input_params)

    @classmethod
    def load_from_cache(cls):
        pass

    @classmethod
    def load_from_db(cls):
        pass