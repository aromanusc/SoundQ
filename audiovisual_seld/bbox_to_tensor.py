import json
import numpy as np
import torch
import glob
import os

import torch 
import torch.nn as nn

dest_path = "/scratch/data/audio-visual-seld-dcase2023/data_dcase2023_task3/video_bboxmat_dev"
bbox_src = "/scratch/data/audio-visual-seld-dcase2023/data_dcase2023_task3/video_bbox_dev"
json_files = glob.glob(os.path.join(bbox_src, '**/*.json'), recursive=True)
class_embeddings = nn.Embedding(50, 8)

for json_file in json_files:
    bbox_data = [] # start the json data
    if "dev-test-sony" not in json_file:
        continue
    with open(json_file, 'r') as f:
        data = json.load(f)
        
        frame_bboxes = []  # To store bounding boxes for this file
        for entry in data:
            pred_boxes = entry.get('pred_boxes', [])  # Extract pred_boxes from the entry
            pred_class = entry.get('pred_classes', [])
            # Initialize a matrix of zeros with 6 rows
            bounding_boxes = np.zeros((6, 12), dtype=np.float32)
            
            # Fill in the bounding box data (up to 6 boxes)
            for i, box in enumerate(pred_boxes[:6]):
                bounding_boxes[i, :4] = box
                class_encode = class_embeddings(torch.tensor(pred_class[i])).detach().cpu().numpy()
                bounding_boxes[i, 4:] = class_encode
                # print(bounding_boxes[i])
                
            frame_bboxes.append(bounding_boxes)
        
        bbox_data.append(frame_bboxes)

        bbox_tensor = np.array(bbox_data)
        bbox_data_name = "/".join(json_file.split("/")[-2:])
        bbox_data_name = bbox_data_name.replace("json", "npy").replace("fold4", "fold5")
        print(bbox_data_name)
        np.save(os.path.join(dest_path, bbox_data_name), bbox_tensor)
