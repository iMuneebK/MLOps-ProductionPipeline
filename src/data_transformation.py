import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import yaml
import pickle

class DataTransformation:
    def __init__(self, config_path="params.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
    def initiate_data_transformation(self):
        print("Starting data transformation...")
        data_path = self.config['data_ingestion']['raw_data_path']
        df = pd.read_csv(data_path)
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['data_transformation']['test_size'], random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save objects
        os.makedirs(os.path.dirname(self.config['data_transformation']['preprocessor_path']), exist_ok=True)
        with open(self.config['data_transformation']['preprocessor_path'], 'wb') as f:
            pickle.dump(scaler, f)
            
        train_arr = pd.concat([pd.DataFrame(X_train_scaled), y_train.reset_index(drop=True)], axis=1)
        test_arr = pd.concat([pd.DataFrame(X_test_scaled), y_test.reset_index(drop=True)], axis=1)
        
        train_arr.to_csv(self.config['data_transformation']['train_data_path'], index=False)
        test_arr.to_csv(self.config['data_transformation']['test_data_path'], index=False)
        print("Data transformation completed.")

if __name__ == "__main__":
    transformer = DataTransformation()
    transformer.initiate_data_transformation()
