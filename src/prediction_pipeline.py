import pickle
import pandas as pd
import yaml

class PredictionPipeline:
    def __init__(self, config_path="params.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
    def predict(self, features):
        model_path = self.config['model_trainer']['model_path']
        preprocessor_path = self.config['data_transformation']['preprocessor_path']
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
            
        scaled_features = preprocessor.transform(features)
        preds = model.predict(scaled_features)
        
        return preds
