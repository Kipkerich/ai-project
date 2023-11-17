from flask import Flask, render_template, request
import sklearn
import numpy as np
import joblib

app= Flask(__name__)


model=joblib.load('regressionModel.pkl')

@app.route('/')
def root():
    return render_template('index.html')


@app.route('/form')
def form():
    return render_template('form.html')


@app.route('/result', methods=['POST','GET'])
def result():
    if request.method == 'POST':
        features = [int(x) for x in request.form.values()]
        final_features = [np.array(features)]
        result = model.predict(final_features)
        return render_template('results.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)