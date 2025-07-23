# iris-api-deploy
Iris Flower Classification – ML API with Docker

A lightweight, production-ready Machine Learning inference API built using FastAPI and Docker. This API classifies Iris flowers based on sepal and petal dimensions using a pre-trained scikit-learn model.

Features

-  REST API built with FastAPI
-  Scikit-learn model trained on the Iris dataset
-  Fully containerized using Docker
-  Reproducible environment with `requirements.txt`
-  Interactive documentation available at `/docs` via Swagger UI



Tech Stack

- Python 3.10+
- FastAPI
- Scikit-learn
- Uvicorn
- Docker



Project Structure:

ml-k8s-deploy/
│
├── app/
│ ├── predict.py # FastAPI app and inference logic
│ ├── model.npy # Serialized ML model
│ └── Dockerfile # Docker build instructions
│
├── requirements.txt # Dependencies
└── README.md # You're here
