import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from flask import Flask, request, render_template

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':

        form_data = {
            'day_type': '',
            'work_hours': '',
            'screen_time_hours': '',
            'meetings_count': '',
            'breaks_taken': '',
            'after_hours_work': '',
            'sleep_hours': '',
            'task_completion_rate': ''
        }

        return render_template('home.html', results=None, form_data=form_data)

    else:

        # Raw form data (for re-populating the form)
        form_data = {
            'day_type': request.form.get('day_type', ''),
            'work_hours': request.form.get('work_hours', ''),
            'screen_time_hours': request.form.get('screen_time_hours', ''),
            'meetings_count': request.form.get('meetings_count', ''),
            'breaks_taken': request.form.get('breaks_taken', ''),
            'after_hours_work': request.form.get('after_hours_work', ''),
            'sleep_hours': request.form.get('sleep_hours', ''),
            'task_completion_rate': request.form.get('task_completion_rate', '')
        }

        try:
            # Convert and validate
            day_type = int(form_data['day_type'])
            task_rate = float(form_data['task_completion_rate'])

            if task_rate < 0 or task_rate > 100:
                raise ValueError("Task completion rate must be between 0 and 100.")

            data = CustomData(
                day_type=day_type,
                work_hours=float(form_data['work_hours']),
                screen_time_hours=float(form_data['screen_time_hours']),
                meetings_count=int(form_data['meetings_count']),
                breaks_taken=int(form_data['breaks_taken']),
                after_hours_work=float(form_data['after_hours_work']),
                sleep_hours=float(form_data['sleep_hours']),
                task_completion_rate=task_rate
            )

            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            predict_pipeline = PredictPipeline()
            print("Mid Prediction")

            results = predict_pipeline.predict(pred_df)

            print("After Prediction")

            return render_template(
                'home.html',
                results=results[0] if results is not None else None,
                form_data=form_data
            )

        except ValueError as e:
            return render_template(
                'home.html',
                results=None,
                form_data=form_data,
                error=str(e)
            )

        except Exception as e:
            return render_template(
                'home.html',
                results=None,
                form_data=form_data,
                error=f"An error occurred: {str(e)}"
            )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
