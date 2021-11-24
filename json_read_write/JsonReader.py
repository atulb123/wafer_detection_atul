import json
class JsonReader:
    def get_value_from_json_file(self,path,key):
        with open(path) as f:
            return json.load(f).get(key)
