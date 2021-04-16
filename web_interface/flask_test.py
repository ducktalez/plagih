from flask import Flask
from flask import jsonify

app = Flask(__name__)


@app.route('/')
def index():
    return f'HEYY!'


@app.route('/rest')
def rest_dummy():
    tdic = {'ticketid': 'qwer',
            'user': 'sfeh1',
            'key3': 'xxx',
            'key4': 'yyy'}
    return jsonify(tdic)


@app.route('/restlist')
def rest_list():
    tdic = [{'ticketid': 'qwer',
             'user': 'sfeh1',
             'key3': 'xxx',
             'key4': 'yyy'}]
    return jsonify(tdic)


if __name__ == '__main__':
    app.run()
