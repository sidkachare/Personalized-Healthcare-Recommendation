from src.data_preprocessing import preprocess_data
from src.model_trainer import train_model
from src.model_evaluation import evaluate_model
from src.inference import run_inference

def main():
    filepath = 'data/raw/healthcare_data.csv'
    target_column = 'Recommended_Treatment'
    X_train, X_test, y_train, y_test, scaler = preprocess_data(filepath, target_column)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    sample = X_test[0].reshape(1, -1)
    prediction = run_inference(sample, model_path='model.pkl', scaler=None)
    print(f"Inference Prediction for sample: {prediction}")

if __name__ == '__main__':
    main()
