"""Temporary helper to scan trees.py structure. Delete after use."""

import re

path = r"C:\Users\Simon\PycharmProjects\plagih\plagih\trees.py"
with open(path, encoding="utf-8") as f:
    lines = f.readlines()
print(f"Total lines: {len(lines)}\n")
print("=== Classes ===")
for i, _l in enumerate(lines, 1):
    if re.match(r"^class ", _l):
        print(f"  {i:5d}: {_l.rstrip()}")
print("\n=== Top-level functions ===")
for i, _l in enumerate(lines, 1):
    if re.match(r"^def ", _l):
        print(f"  {i:5d}: {_l.rstrip()}")
print("\n=== Section markers ===")
for i, _l in enumerate(lines, 1):
    if re.match(r"^# =+", _l):
        print(f"  {i:5d}: {_l.rstrip()}")
