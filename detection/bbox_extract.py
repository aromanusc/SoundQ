import torch
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
# import some common libraries
import sys
import glob
import numpy as np
import os, json, cv2, random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test


BUILDIN_CLASSIFIER = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

from detic.modeling.text.text_encoder import build_text_encoder

def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

class BoundingBoxExtractor:
    def __init__(self, config_file, weights_url):
        self.cfg = get_cfg()
        add_centernet_config(self.cfg)
        add_detic_config(self.cfg)
        self.cfg.merge_from_file(config_file)
        self.cfg.MODEL.WEIGHTS = weights_url
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
        self.cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        self.cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True

        self.predictor = DefaultPredictor(self.cfg)

        self.metadata = None
        self.classifier = None

        self.transform_needs_flip = ["aug_acs_1", "aug_acs_4", "aug_acs_5", "aug_acs_8"]

    def load_custom_vocabulary(self):
        custom_vocabulary = ["person", "human", "mouth", "head", "headband", "hat",
                            "hands", "telephone", "cellular", "computer", "laptop",
                            "vaccum", "broom", "mop", "stool", "fan", "microwave", "coffee table", "sink", "dishes", "microwave", "water boiler", "kettle",
                            "shoes", "door", "knob", "cup board", "tea tray", "tray", "drawer", "drawer", "pot",
                            "loud speaker", "speaker",
                            "shoes", "instrument", "guitar", "musical instrument", "Marimba", "xylophone", "piano", "cowbell", "rattle",
                            "water tap", "faucet", "dish washer", "sink",
                            "bell", "glass cup", "cup"]

        self.metadata = MetadataCatalog.get("__unused")
        self.metadata.thing_classes = custom_vocabulary

        self.classifier = get_clip_embeddings(custom_vocabulary)
        num_classes = len(custom_vocabulary)
        reset_cls_test(self.predictor.model, self.classifier, num_classes)

    def process_video(self, video_file, dest_path):
        video_capture = cv2.VideoCapture(video_file)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_data = []
        for frame_idx in range(frame_count):
            ret, frame = video_capture.read()
            if not ret:
                break

            if not all(flip_file not in video_file for flip_file in self.transform_needs_flip):
                frame = np.flip(frame, axis=0) # flip across elevation axis

            outputs = self.predictor(frame)
            # Convert outputs["instances"] to a dictionary and save it as JSON
            instances_per_frame = {
                'pred_classes': outputs["instances"].pred_classes.cpu().tolist(),
                'pred_class_names': [self.metadata.thing_classes[x] for x in outputs["instances"].pred_classes.cpu().tolist()],
                'scores': outputs["instances"].scores.cpu().tolist(),
                'pred_boxes': outputs["instances"].pred_boxes.tensor.cpu().tolist()
            }
            frame_data.append(instances_per_frame)
            # Uncomment to visualize independent frames
            # v = Visualizer(frame[:, :, ::-1], self.metadata)
            # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # cv2.imwrite(f"image_{frame_idx}.jpg", out.get_image()[:, :, ::-1])
        video_capture.release()
        frame_data_name = "/".join(video_file.split("/")[-2:])
        frame_data_name = frame_data_name.replace("mp4", "json")
        with open(os.path.join(dest_path, frame_data_name), 'w') as f:
            json.dump(frame_data, f, indent=4)


if __name__ == "__main__":
    config_file = "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
    weights_url = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'

    extractor = BoundingBoxExtractor(config_file, weights_url)
    extractor.load_custom_vocabulary()

    video_dev_src = "/scratch/data/audio-visual-seld-dcase2023/data_dcase2023_task3/video_dev/"  # Provide the path to the input video file
    dest_path = "/scratch/data/audio-visual-seld-dcase2023/data_dcase2023_task3/video_bbox_dev"
    mp4_file_paths = ["/scratch/data/audio-visual-seld-dcase2023/data_dcase2023_task3/video_dev/dev-train-tau-aug-acs/fold3_room6_mix005_aug_acs_4.mp4"]#glob.glob(os.path.join(video_dev_src, '**/*.mp4'), recursive=True)
    total_files = len(mp4_file_paths)
    for i, video_file in enumerate(mp4_file_paths):
        print(f"File progress {i+1}/{total_files}")
        extractor.process_video(video_file, dest_path)
