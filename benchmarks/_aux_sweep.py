# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import json

DO_SLEEP_BETWEEN = False

if os.path.exists("do_sleep.json"):
    with open("do_sleep.json", "r", encoding="utf-8") as f:
        dict_ = json.load(f)
    DO_SLEEP_BETWEEN = bool(dict_["do_sleep"])
