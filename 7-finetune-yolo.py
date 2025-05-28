import os
import json
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path

def train_yolo_seg():
    try:
        # Training configuration
        name = "v4-8n-seg"
        yaml_path = 'yolo_split/data.yaml'
        epochs = 45
        imgsz = 512
        batch = 16
        patience = 20
        workers = 4
        save_period = 5
        lr0 = 0.005
        device = 'cuda'
        
        # Verify dataset paths
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Dataset YAML file not found at {yaml_path}")
            
        # Define paths
        save_dir = f"models/{name}"
        os.makedirs(save_dir, exist_ok=True)

        # Define training parameters
        train_params = {
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'patience': patience,
            'device': device,
            'workers': workers,
            'save_period': save_period,
            'lr0': lr0,
        }

        # Save training parameters
        with open(os.path.join(save_dir, 'train_params.json'), 'w') as f:
            json.dump(train_params, f, indent=4)

        # Load the model
        model = YOLO('yolo8n-seg.pt')

        # Fine-tune the model
        results = model.train(
            data=str(yaml_path),
            epochs=train_params['epochs'],
            imgsz=train_params['imgsz'],
            batch=train_params['batch'],
            patience=train_params['patience'],
            save=True,
            save_period=train_params['save_period'],
            device=train_params['device'],
            workers=train_params['workers'],
            project=save_dir,
            name=name,
            pretrained=True,
            lr0=train_params['lr0'],
            cos_lr=False,
            close_mosaic=10,
            amp=True,
            overlap_mask=True,
            conf=0.5,
            seed=42
        )

        # Save training metadata
        metadata = {
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'base_model': 'yolo8n-seg.pt',
            'parameters': train_params,
            'results': {
                'metrics': results.results_dict if hasattr(results, 'results_dict') else {},
                'best_map': results.best_map if hasattr(results, 'best_map') else None,
                'best_epoch': results.best_epoch if hasattr(results, 'best_epoch') else None,
                'metrics_keys': results.keys if hasattr(results, 'keys') else [],
                'metrics_vals': results.values if hasattr(results, 'values') else []
            }
        }

        # Save all available metrics
        if hasattr(results, 'results_dict'):
            for key, value in results.results_dict.items():
                if isinstance(value, (list, tuple)):
                    metadata['results'][f'{key}_history'] = value

        with open(os.path.join(save_dir, 'training_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Training completed. Model and metadata saved in '{save_dir}' directory")
        print("\nAvailable metrics:", metadata['results']['metrics_keys'] if 'metrics_keys' in metadata['results'] else "No metrics available")

    except Exception as e:
        print(f"Error during training: {e}")
        raise  # Re-raise the exception for better error tracking

if __name__ == "__main__":
    train_yolo_seg() 