import os
import pickle
import yaml
from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    def __init__(self, config_path="params.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
    def initiate_model_training(self):
        print("Starting model training...")
        train_data_path = self.config['data_transformation']['train_data_path']
        
        import pandas as pd
        train_df = pd.read_csv(train_data_path)
        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        os.makedirs(os.path.dirname(self.config['model_trainer']['model_path']), exist_ok=True)
        with open(self.config['model_trainer']['model_path'], 'wb') as f:
            pickle.dump(model, f)
            
        print("Model training completed and saved.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_training()
