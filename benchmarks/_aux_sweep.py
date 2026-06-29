import os
import json

DO_SLEEP_BETWEEN = False

if os.path.exists("do_sleep.json"):
    with open("do_sleep.json", "r", encoding="utf-8") as f:
        dict_ = json.load(f)
    DO_SLEEP_BETWEEN = bool(dict_["do_sleep"])
