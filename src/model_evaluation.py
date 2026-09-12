import pandas as pd
import pickle
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelEvaluation:
    def __init__(self, config_path="params.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
    def evaluate(self):
        print("Starting model evaluation...")
        test_data_path = self.config['data_transformation']['test_data_path']
        model_path = self.config['model_trainer']['model_path']
        
        test_df = pd.read_csv(test_data_path)
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        mlflow.set_experiment("mlops-pipeline-experiment")
        with mlflow.start_run():
            predictions = model.predict(X_test)
            
            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions)
            rec = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            
            mlflow.sklearn.log_model(model, "model")
            
            print(f"Model Evaluation Metrics: Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.evaluate()
