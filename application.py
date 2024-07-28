from flask import Flask, request, render_template
from src.pipelines.prediction_pipeline import GetDataframe, GetPrediction

application = Flask(__name__)

app = application


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'POST':

        input_data = GetDataframe(
            carat=float(request.form.get('carat')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')
        )

        df_input = input_data.make_dataframe()
        pred_obj = GetPrediction()
        val_pred = pred_obj.make_prediction(df_input)

        result = round(val_pred[0], 2)

        return render_template('result.html', final_result=result)

    else:
        return render_template('form.html')


if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
