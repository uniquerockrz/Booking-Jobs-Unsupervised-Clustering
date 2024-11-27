# Booking Jobs Unsupervised Classification

This repo contains code for doing unsupervised classification on the Booking.com Jobs Dataset.
It has notebook where EDA has been done and the model has been built, as well as the models and required artifacts. 

It also has a FastAPI server which exposes a REST API to do predictions on the model. 

## The Data

The data can be found at [Kaggle](https://www.kaggle.com/code/niekvanderzwaag/booking-com-jobs-eda-nlp-ensemble-modeling). Download the data and put it inside the `data` folder.

## Setup

You can use the provided `requirements.txt` file to install the requirements inside a virtual environment. 

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Building The Model

Prebuilt models and artifacts can be found inside the `models` folder. Or you can also look into `Notebooks/eda.ipynb' file to build the model yourself. 

## Starting The FastAPI Server

You can use the following command to start the API server.

```bash
uvicorn app:app
```

## Using The API To Make Predictions

Use the following Python code to send a POST request to `/predict` with the payload. 

```python
url = "http://localhost:8000/predict/"

payload = {
    "text": "Paste Job Description Here."
}

response = requests.post(url, json=payload)
print(response.text)
```

Docs can be found at the `/docs` endpoint. 

## Using Docker

Run the docker to start the FastAPI server. 

```bash
docker compose up
```