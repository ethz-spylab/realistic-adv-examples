import json
import os
from pathlib import Path


class JSONList:
    def __init__(self, path: Path) -> None:
        self.path: Path = path
    
    def append(self, item: list[dict]) -> None:
        if not self.path.exists():
            first_element = True
            with self.path.open("w") as f:
                json.dump([], f)
        else:
            first_element = False
        item_as_json = json.dumps(item)
        if first_element:
            # We should not put a comma if it is the first element
            string_to_write = f"{item_as_json}]"
        else:
            string_to_write = f",{item_as_json}]"
        # Delete last character ("]") and append the new item
        with self.path.open("rb+") as f:
            f.seek(-1, os.SEEK_END)
            f.truncate()
        with self.path.open("a") as f:
            f.write(string_to_write)
