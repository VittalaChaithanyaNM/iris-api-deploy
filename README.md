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

Live API Demo

🔗 [https://iris-api-deploy.onrender.com/docs](https://iris-api-deploy.onrender.com/docs)  
→ Swagger UI for testing endpoints


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

 Setup Instructions
1. Clone the repository
  git clone https://github.com/VittalaChaithanyaNM/iris-api-deploy.git
  cd iris-api-deploy

2.Create a virtual environemet to run this clones repo
  python -m venv venv
  source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install the dependecies
   pip install -r requirements.txt

4.Run the server locally
  uvicorn main:app --reload
Visit http://127.0.0.1:8000/docs to open the Swagger UI and test the API locally.


