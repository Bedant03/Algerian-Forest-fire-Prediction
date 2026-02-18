import pickle
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template

app = Flask(__name__)

static_path = os.path.join(app.root_path, "static")
if not os.path.exists(static_path):
    os.makedirs(static_path)

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            BUI = float(request.form.get('BUI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            input_data = [[
                Temperature, RH, Ws, Rain,
                FFMC, DMC, ISI, BUI,
                Classes, Region
            ]]

            scaled_data = standard_scaler.transform(input_data)
            prediction = ridge_model.predict(scaled_data)
            predicted_fwi = round(prediction[0], 2)

            if predicted_fwi < 5:
                risk_level = "Low Risk"
            elif predicted_fwi < 15:
                risk_level = "Moderate Risk"
            else:
                risk_level = "High Risk"

            features = ['Temp', 'RH', 'Ws', 'Rain',
                        'FFMC', 'DMC', 'ISI', 'BUI']
            values = [Temperature, RH, Ws, Rain,
                      FFMC, DMC, ISI, BUI]

            plt.figure(figsize=(8, 4))
            plt.bar(features, values)
            plt.xticks(rotation=45)
            plt.title("Input Weather Conditions")
            plt.tight_layout()
            plt.savefig(os.path.join(static_path, "input_plot.png"))
            plt.close()

            return render_template(
                "index1.html",
                results=predicted_fwi,
                risk=risk_level,
                plot_url="input_plot.png"
            )

        except Exception as e:
            print("ERROR:", e)
            return render_template("index1.html", results="Error occurred")

    return render_template("index1.html")

@app.route("/dashboard")
def dashboard():

    df = pd.read_csv("Algerian_forest_fires_cleaned_dataset.csv")

    plt.figure()
    df['Classes'].value_counts().plot(kind='bar')
    plt.title("Fire vs Not Fire Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(static_path, "class_distribution.png"))
    plt.close()

    plt.figure()
    plt.scatter(df['Temperature'], df['FWI'])
    plt.xlabel("Temperature")
    plt.ylabel("FWI")
    plt.title("Temperature vs FWI")
    plt.tight_layout()
    plt.savefig(os.path.join(static_path, "temp_vs_fwi.png"))
    plt.close()

    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(static_path, "heatmap.png"))
    plt.close()
    plt.figure()
    df['FWI'].plot(kind='hist', bins=20)
    plt.title("FWI Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(static_path, "fwi_hist.png"))
    plt.close()

    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)