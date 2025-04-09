# Personalized Healthcare Recommendation System

This project builds a machine learning pipeline to predict healthcare outcomes based on patient features. It includes data preprocessing, model training, evaluation, and serving via a FastAPI endpoint.

## Project Structure
   ### Personalized Healthcare Recommendation
        --data/
            --raw
                --healthcare_data.csv
        
        --notebook
            --EDA.ipynb
        
        --src
            --data_preprocessing.py
            --model_trainer.py
            --model_evaluation.py
            --inference.py
        
        --app.py

        --main.py

        --model.pkl

        --requirements.txt

## Create a Virtual Environment

## Install Dependencies
    pip install -r requirements.txt

## Run the ML Pipeline
    python main.py

## Inference with FastAPI
 ## Run the API:
    uvicorn app:app --reload


