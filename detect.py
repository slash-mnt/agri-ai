import cv2
import os
import supervision as sv
import numpy as np
from ultralytics import YOLO

images_path = "./images/"
results_path = "./results/"

files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

model = YOLO("./runs/detect/train2/weights/best.pt")
#model = YOLO("yolov8n.pt")

detected = 0
with sv.ImageSink(target_dir_path=results_path, overwrite=True) as sink:
    for file in files:
        image = cv2.imread(images_path + file)
        results = model(image)[0]

        def callback(image_slice: np.ndarray) -> sv.Detections:
            result = model(image_slice)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > 0.5]
            return detections

        slicer = sv.InferenceSlicer(callback = callback)
        detections = slicer(image)

        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        percentage_annotator = sv.PercentageBarAnnotator();

        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        #annotated_image = label_annotator.annotate(scene=image.copy(), detections=detections)
        #annotated_image = percentage_annotator.annotate(scene=image.copy(), detections=detections)

        sink.save_image(image=annotated_image, image_name=file)
        detected += len(detections[detections.class_id == 0])

print("Detected %d haystacks in the dataset" % detected)
