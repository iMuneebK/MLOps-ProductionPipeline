import pandas as pd
import numpy as np
import os
import yaml

class DataIngestion:
    def __init__(self, config_path="params.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)['data_ingestion']

    def initiate_data_ingestion(self):
        """Simulates data ingestion from a source."""
        print("Starting data ingestion...")
        # Create dummy data for the sake of a portfolio project
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.rand(1000),
            'feature2': np.random.rand(1000) * 10,
            'feature3': np.random.randint(0, 100, 1000),
            'target': np.random.randint(0, 2, 1000)
        })
        
        os.makedirs(os.path.dirname(self.config['raw_data_path']), exist_ok=True)
        data.to_csv(self.config['raw_data_path'], index=False)
        print(f"Data saved to {self.config['raw_data_path']}")
        return self.config['raw_data_path']

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()
