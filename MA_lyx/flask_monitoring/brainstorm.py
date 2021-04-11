from flask import Flask, jsonify


app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    testdir = {
        'asd': 'asd',
        'hello:': 'world',
    }
    return f"{testdir}"


app.run()
