import torch

# Function to download YOLOv8 model

def download_yolov8_model(model_name='yolov8s'):
    """
    Download the specified YOLOv8 model.

    Parameters:
    model_name (str): Name of the YOLOv8 model to download.
    """
    model_url = f'https://github.com/ultralytics/yolov8/releases/download/v0.0.1/{model_name}.pt'
    model_path = f'./{model_name}.pt'
    torch.hub.download_url_to_file(model_url, model_path)
    return model_path

if __name__ == '__main__':
    model_path = download_yolov8_model()
    print(f'Model downloaded to: {model_path}')