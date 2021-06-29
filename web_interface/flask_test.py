from flask import Flask
from flask import jsonify

app = Flask(__name__)


@app.route('/')
def index():
    return f'HEYY!'


@app.route('/rest')
def rest_dummy():
    tdic = {'object_id': '0091000335',
            'person_resp': 'sfehrer',
            'created_by': 'utoe',
            'description': 'normaler Report 01',
            'created_at': '20201026145909',
            'concatstat': 'Offen',
            'valid_from': '20201026',
            'priority': '4',
            'aufwand_plan_d': '0 ',
            'aufwand_ist_d': '0.00 '}
    return jsonify(tdic)


@app.route('/restlist')
def rest_list():
    tdic = [{'object_id': '0091000335',
             'person_resp': 'sfehrer',
             'created_by': 'utoe',
             'description': 'normaler Report 01',
             'created_at': '20201026145909',
             'concatstat': 'Offen',
             'valid_from': '20201026',
             'priority': '4',
             'aufwand_plan_d': '0 ',
             'aufwand_ist_d': '0.00 '},
            {'object_id': '0001',
             'person_resp': 'fheg',
             'created_by': 'alam',
             'description': 'normaler Report 01',
             'created_at': '20211026145909',
             'concatstat': 'Offen',
             'valid_from': '20201126',
             'priority_txt': '4: Niedrig',
             'aufwand_plan_d': '0 ',
             'aufwand_ist_d': '0.00 '}]
    return jsonify(tdic)


if __name__ == '__main__':
    app.run()
