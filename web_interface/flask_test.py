from flask import Flask


app = Flask(__name__)


@app.route('/')
def index():
    return f'HEYY!'


@app.route('/rest')
def rest_example():
    tdic = {'ticket_id': 'qwer',
            'user': 'sfeh1',
            'key3': 'xxx',
            'key4': 'yyy'}

    return tdic


if __name__ == '__main__':
    app.run()
