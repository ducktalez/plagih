"""
#!/usr/bin/env bash
pdflatex agents_trees.tex
find . -name "visualisation/*.tex" -exec pdflatex {} \;
find . -name "visualisation/*.pdf" -exec pdftoppm {} {} -png \;
a*b * Ifte ** a*b*c *I *Ifte
"""


if __name__ == '__main__':
    # import requests
    # import json
    # from pathlib import Path
    # response = requests.get('http://localhost:5000/restlist')
    # pools = json.loads(response.content)
    # txt_result = ''
    # for row in pools:
    #     txt_drawer = f"\n" \
    #                  f"** sfeh:ALAM {row['description']}\n" \
    #                  f"   :PROPERTIES:\n" \
    #                  f"   :SOLMAN:   {row['object_id']}\n" \
    #                  f"   :EFFORT:   {row['aufwand_plan_d']}d\n" \
    #                  f"   :CREATED:  {row['created_by']}\n" \
    #                  f"   :STATUS:   {row['concatstat']}\n" \
    #                  f"   :allocate: {row['person_resp']}\n" \
    #                  f"   :IST:      {row['aufwand_ist_d']}d\n" \
    #                  f"   :END:\n"
    #     txt_result += txt_drawer
    #
    # with Path('C:/Users/Rapid/Desktop/bru-projekte/pytest.org').open('w') as file:
    #     file.write(txt_result)
    pass
