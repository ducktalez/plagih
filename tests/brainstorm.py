dct = {1: {'a': 10, 'b': 4},
       3: {'a': 30, 'b': 3},
       2: {'a': 40, 'b': 2},
       4: {'a': 20, 'b': 1}}

import json
from pathlib import Path

(Ifte((Or(((observation1)<(1)), (((observation1)<(0.1))&((observation0)<(-0.05))))), (2), (Ifte((((observation0)<(0.02))&(((observation1)>(-0.45))&((observation1)<(-0.05)))), (0), (Ifte(((observation0)<(0)), (0), (2)))))))