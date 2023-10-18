import re
import sympy

x = """
    0000 : 62 50 00 32 38 37 39 37   37 32 30 31 00 0C 07 77  bP.28797  7201...w
    0010 : 00 00 04 FF FF FF FF FF   FF FF FF FF 00 0C 07 77  ...ÿÿÿÿÿ  ÿÿÿÿ...w
    0020 : 00 00 18 FF FF FF FF FF   FF FF FF FF 00 13 0C 76  ...ÿÿÿÿÿ  ÿÿÿÿ...v
    0030 : 00 00 83 FF FF FF FF FF   FF FF FF FF 00 0C 07 77  ..?ÿÿÿÿÿ  ÿÿÿÿ...w
"""
a = x.split('\n')
a = a[1:5]  # remove empty line 1 and line
for row in a:
    row = row[11:60]  # alternative: regex-search .*(:\s)
    row = re.split(r'\W', row)
    row = row[:8] + row[-8:]
    row = [str(int(x, 16)) for x in row]
    row = '[[' + ', '.join(row[:8]) + '], [' + ', '.join(row[-8:]) + ']]'
    # row = row.replace('   ', '],[')
    # row = row.replace(' ', ', ')
    # row = '[[' + row + ']]'
    print(row)

"""[[98, 80, 0, 50, 56, 55, 57, 55], [55, 50, 48, 49, 0, 12, 7, 119]]
[[0, 0, 4, 255, 255, 255, 255, 255], [255, 255, 255, 255, 0, 12, 7, 119]]
[[0, 0, 24, 255, 255, 255, 255, 255], [255, 255, 255, 255, 0, 19, 12, 118]]
[[0, 0, 131, 255, 255, 255, 255, 255], [255, 255, 255, 255, 0, 12, 7, 119]]"""

sympy.sympify('round(1.23)')
