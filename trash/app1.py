import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

from flask import Flask, request, render_template

appplication = Flask(__name__)

app = appplication

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        # return render_template('home.html')
        return render_template('home.html', results=None, form_data={})
    else:
        # data=CustomData(
        #     gender=request.form.get('gender'),
        #     race_ethnicity=request.form.get('ethnicity'),
        #     parental_level_of_education=request.form.get('parental_level_of_education'),
        #     lunch=request.form.get('lunch'),
        #     test_preparation_course=request.form.get('test_preparation_course'),
        #     reading_score=float(request.form.get('writing_score')),
        #     writing_score=float(request.form.get('reading_score'))

        # )

        # data = CustomData(
        #     gender=request.form.get('gender'),
        #     race_ethnicity=request.form.get('ethnicity'),
        #     parental_level_of_education=request.form.get('parental_level_of_education'),
        #     lunch=request.form.get('lunch'),
        #     test_preparation_course=request.form.get('test_preparation_course'),
        #     reading_score=float(request.form.get('reading_score')),   # ✅ CORRECT
        #     writing_score=float(request.form.get('writing_score'))    # ✅ CORRECT
        # )

        # pred_df=data.get_data_as_data_frame()
        # print(pred_df)
        # print("Before Prediction")

        # predict_pipeline=PredictPipeline()
        # print("Mid Prediction")
        # results=predict_pipeline.predict(pred_df)
        # print("after Prediction")
        # return render_template('home.html',results=results[0])

        # Capture RAW form data BEFORE processing (for repopulation)
        form_data = {
            'gender': request.form.get('gender', ''),
            'ethnicity': request.form.get('ethnicity', ''),
            'parental_level_of_education': request.form.get('parental_level_of_education', ''),
            'lunch': request.form.get('lunch', ''),
            'test_preparation_course': request.form.get('test_preparation_course', ''),
            'writing_score': request.form.get('writing_score', ''),
            'reading_score': request.form.get('reading_score', '')
        }
        
        # FIX: Use CORRECT score mappings here (see Step 1 above)
        data = CustomData(...)  # With corrected scores
        
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Pass BOTH results AND form data to template
        return render_template(
            'home.html', 
            results=results[0] if results else None,
            form_data=form_data
        )

if __name__=="__main__":
    # app.run(host="0.0.0.0")    
    app.run(debug=True, port=8000) # port:optional, for other  
    # app.run(debug=False, use_reloader=False, port=8000)