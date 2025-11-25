import roboflow
import dotenv
import os

class DataLoader:
    def __init__(self, config):
        dotenv.load_dotenv()
        self.config = config
        self.rf = roboflow.Roboflow(api_key=os.getenv("API_KEY"))
        self.project = self.rf.workspace(config['roboflow']['workspace']).project(config['roboflow']['project'])
        self.version = self.project.version(config['roboflow']['version'])
        self.model_format = config['roboflow']['model_format']

    def download_dataset(self):
        version = self.project.version(self.version)
        dataset = self.rf.download_dataset(
            dataset_url="https://app.roboflow.com/ds/mjH83h7Mis?key=QtuvKb0qEs",
            model_format=self.model_format,
            location="./data"
            )
        return dataset