from flask import Flask


app = Flask(__name__)


@app.route('/')
def index():
    return f'HEYY!'


@app.route('/rest')
def rest_example():
    tdic = {'key1': 'qwer',
            'key2': {'subkey1': 'xxx',
                     'subkey2': 'yyy'}}

    return tdic


if __name__ == '__main__':
    app.run()
