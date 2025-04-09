from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train, model_path='model.pkl'):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model
