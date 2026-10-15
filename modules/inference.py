from PIL import Image
import supervision as sv


def run_inference(img, model, debug=False):
    results = model.infer(img)[0]
    detections = sv.Detections.from_inference(results)

    if debug:
        from datetime import datetime
        import os
        import numpy as np

        output_dir = "images/annotated_images"
        os.makedirs(output_dir, exist_ok=True)
        str_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        file_name = f"annotated_{str_time}.png"
        full_path = os.path.join(output_dir, file_name)

        annotated = sv.BoxAnnotator().annotate(scene=img, detections=detections)
        annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections)
        if isinstance(annotated, np.ndarray):
            annotated = Image.fromarray(annotated)
        annotated.save(full_path)
        print(f"saved {full_path}")

    return detections
