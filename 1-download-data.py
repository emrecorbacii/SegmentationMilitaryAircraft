# pip install segments-ai
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
import cv2
import numpy as np

# Initialize a SegmentsDataset from the release file
client = SegmentsClient('73c095db93b91d54e283c812cf6e072ff55e764c')
release = client.get_release('sezaiufuk/AIRCRAFT_GROUPED', 'v1') # Alternatively: release = 'flowers-v1.0.json'
dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['reviewed'])

# Export to YOLO format
export_dataset(dataset, export_format='coco-instance')

