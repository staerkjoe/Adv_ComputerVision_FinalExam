from pathlib import Path
import torch
import wandb
import sys

class YoloModel:
    def __init__(self, config):
        self.config = config
        self.model_name = self.config['model']['name']
        self.model_format = self.config['roboflow']['model_format']
        self.model = None
        self.project_name = self.config['wandb']['project']
        self.run_name = self.config['wandb']['run_name']
        self.artifact_name = self.config['wandb']['artifact_name']
        self.model_filename = self.config['wandb']['model_filename']

    def load_model(self, device=None):
        """Load YOLO model and keep reference in self.model."""
        if self.model_format == "yolov8":
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            if device:
                # optional: move underlying torch model to device
                try:
                    self.model.to(device)
                except Exception:
                    pass
            return self.model
        else:
            raise ValueError(f"Unsupported model format: {self.model_format}")
        
    def load_yolo_from_wandb(self):
        from ultralytics import YOLO
        
        print(f"Loading YOLO model from W&B project: {self.project_name}")
        print(f"Artifact: {self.artifact_name}")
        
        # Initialize W&B and download artifact
        with wandb.init(project=self.project_name, job_type="inference") as run:
            # Download the artifact
            artifact = run.use_artifact(self.artifact_name)
            artifact_dir = artifact.download()
            
            # Get the model weights path
            weights_path = Path(artifact_dir) / self.model_filename
            print(f"Loading YOLO weights from: {weights_path}")
            
            # Load YOLO model (Ultralytics handles everything)
            model = YOLO(weights_path)
            
            print(f"YOLO model loaded successfully")
            return model


    