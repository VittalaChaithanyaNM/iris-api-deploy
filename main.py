from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Load model parameters
weights = np.load("weights.npy")
biases = np.load("biases.npy")

app = FastAPI()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

@app.post("/predict")
def predict_species(data: IrisInput):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    logits = np.dot(features, weights) + biases
    probs = softmax(logits)
    prediction = int(np.argmax(probs))
    class_names = ["setosa", "versicolor", "virginica"]
    return {"class": class_names[prediction], "confidence": round(float(np.max(probs)), 3)}

