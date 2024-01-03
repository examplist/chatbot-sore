from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/", methods=["GET"])
def main():
    response = {"version": "2.0"}
    return jsonify(response)


app.run("0.0.0.0", debug=True, port=5000)
