from flask import Flask


app = Flask(__name__)


@app.route('/')
def index():
    return f'HEYY!'


@app.route('/rest')
def rest_example():
    tdic = {'key1': 'qwer',
            'key2': 'tzui',
            'key3': 'xxx',
            'key4': 'yyy'}

    return tdic


if __name__ == '__main__':
    app.run()
