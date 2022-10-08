from sys import stderr

from flask import Flask , render_template , request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl" , "rb"))

col = ["lotsize" , "bedrooms" , "bathrms"]

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route("/predict" , methods=["POST" , "GET"])
def predict():
    size = request.form.get("size")
    bed = request.form.get("bed")
    bath = request.form.get("bath")
    int_features = [size,bed,bath]
    final = np.array([int_features],dtype=float)
    output = model.predict(final)
    final_output = round(output[0],1)

    return render_template("index.html", pred="The Price of your house is {} thousand dollar".format(final_output))

if __name__ == '__main__':
    app.run(debug=True)