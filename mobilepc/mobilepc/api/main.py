import os
import subprocess
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from predictor.predict import ModelPredictor
from starlette.responses import JSONResponse

from .models.models import Mobile

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)


app = FastAPI()


@app.get('/', status_code=200)
async def healthcheck():
    return 'Mobile classifier is all ready to go!'


@app.post('/predict')
def extract_name(mobile_features: Mobile):
    try:
        predictor = ModelPredictor(
            "C:/Users/IOR_C/OneDrive/Documentos/GitHub/mlops-mobilepc/mobilepc/mobilepc/models/SVM_output.pkl")
        X = {'battery_power': [mobile_features.battery_power],
             'blue': [mobile_features.blue],
             'clock_speed': [mobile_features.clock_speed],
             'dual_sim': [mobile_features.dual_sim],
             'fc': [mobile_features.fc],
             'four_g': [mobile_features.four_g],
             'int_memory': [mobile_features.int_memory],
             'm_dep': [mobile_features.m_dep],
             'mobile_wt': [mobile_features.mobile_wt],
             'n_cores': [mobile_features.n_cores],
             'pc': [mobile_features.pc],
             'px_height': [mobile_features.px_height],
             'px_width': [mobile_features.px_width],
             'ram': [mobile_features.ram],
             'sc_h': [mobile_features.sc_h],
             'sc_w': [mobile_features.sc_w],
             'talk_time': [mobile_features.talk_time],
             'three_g': [mobile_features.three_g],
             'touch_screen': [mobile_features.touch_screen],
             'wifi': [mobile_features.wifi]}
        prediction = predictor.predict(pd.DataFrame(X))
        return JSONResponse(f"Predicted Price Range: {prediction}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "result: invalid format",
                "hints": [
                    "Check the numbers",
                    "Must be float",
                    "It is recommended avoid all zeros in petition"
                ],
            },
        ) from e


@app.post("/train-model/")
async def train_model():
    try:
        result = subprocess.run(
            ["C:/Users/IOR_C/OneDrive/Documentos/GitHub/mlops-mobilepc/venv/Scripts/python",
             "C:/Users/IOR_C/OneDrive/Documentos/GitHub/mlops-mobilepc/mobilepc/mobilepc/mobilepc.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        if result.returncode == 0:
            message = "Model training script executed successfully"
            response_text = result.stdout
        else:
            message = f"An error occurred: {result.stderr}"
            response_text = result.stderr

        response_data = {
            "message": message,
            "response_text": response_text
        }

        return JSONResponse(content=response_data)
    except subprocess.CalledProcessError as e:
        error_message = f"An error occurred: {e}\n{e.stderr}"
        return JSONResponse(content={"error": error_message}, status_code=500)
