from flask import Flask, render_template, request
from inference import predict_sentiment

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        sentiment = predict_sentiment(text)
        return render_template("index.html", text=text, sentiment=sentiment)

    return render_template("index.html", text="", sentiment="")

if __name__ == "__main__":
    app.run(debug=True)
