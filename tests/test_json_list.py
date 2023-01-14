import json
from pathlib import Path

from src.json_list import JSONList

def test_json_list(tmp_path: Path):
    path = tmp_path / "test.json"
    json_list = JSONList(path)
    object_to_write_1 = [{"test1": "test1"}]
    json_list.append(object_to_write_1)
    with path.open("r") as f:
        assert json.load(f) == [object_to_write_1]
    object_to_write_2 = [{"test2": "test2"}]
    json_list.append(object_to_write_2)
    with path.open("r") as f:
        assert json.load(f) == [object_to_write_1, object_to_write_2]