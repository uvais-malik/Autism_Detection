
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and encoders once
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Extract dropdown options
countries = encoders['Country of residence'].classes_.tolist()
ethnicities = encoders['ethnicity'].classes_.tolist()
relations = encoders['relation'].classes_.tolist()

# Mapping of answer choices to binary scores
answer_map = {
    "definitely agree": 1,
    "slightly agree": 1,
    "slightly disagree": 0,
    "definitely disagree": 0
}

@app.route('/')
def index():
    return render_template('form.html', countries=countries, ethnicities=ethnicities, relations=relations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert A1–A10 answers from radio values to 0 or 1 using mapping
        aq_scores = {}
        for i in range(1, 11):
            q_key = f"A{i}"
            answer = request.form[q_key].strip().lower()
            aq_scores[f"A{i}_Score"] = answer_map.get(answer, 0)

        # Collect the rest of the form data
        input_data = {
            **aq_scores,
            'age': int(request.form['age']),
            'gender': request.form['gender'],
            'ethnicity': request.form['ethnicity'],
            'jaundice': request.form['jaundice'],
            'austism': request.form['austism'],
            'Country of residence': request.form['Country of residence'],
            'used_app_before': request.form['used_app_before'],
            'relation': request.form['relation']
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Encode categorical columns
        for col in ['gender', 'ethnicity', 'jaundice', 'austism', 'Country of residence', 'used_app_before', 'relation']:
            df[col] = encoders[col].transform(df[col])

        # Reorder columns to match training data
        df = df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                 'age', 'gender', 'ethnicity', 'jaundice', 'austism',
                 'Country of residence', 'used_app_before', 'relation']]

        # Make prediction
        prediction = model.predict(df)[0]
        result = "Autism Detected" if prediction == 1 else "No Autism Detected"

        return render_template('form.html', result=result, countries=countries, ethnicities=ethnicities, relations=relations)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
