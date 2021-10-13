import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm

from experiments.detect import R, Representer
from experiments.track import Tracker


def get_images(data_dir: Path) -> Iterable[np.ndarray]:
    with tqdm(list(sorted(data_dir.joinpath("img1").glob("*.jpg")))) as pbar:
        for img in pbar:
            pbar.desc = img.name
            yield np.array(Image.open(img).convert("RGB")) / 255


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Path to the dataset", type=Path)
    parser.add_argument(
        "--output_file", "-o", help="File to save results in", default="results.pickle"
    )
    parser.add_argument("--checkpoint", help="Path to SSDIR checkpoint", type=Path)
    parser.add_argument(
        "--confidence_threshold",
        help="Confidence threshold to be used for evaluation",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--nms_threshold",
        help="NMS threshold to be used for evaluation",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--max_per_image",
        help="Max number of detections per image",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    detector = Representer(
        checkpoint=args.checkpoint,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        max_per_image=args.max_per_image,
    )

    with tqdm(list(sorted(args.data_dir.glob("*/")))) as pbar:
        for data_dir in pbar:
            pbar.desc = data_dir.name
            output_path = data_dir / args.output_file
            data = {}
            with output_path.open("wb") as fp:
                for frame_idx, image in enumerate(get_images(data_dir), start=1):
                    data[frame_idx] = list(detector(image))
                pickle.dump(data, fp)
                # for frame_idx, objects in enumerate(tracker(detector(image) for image in get_images(data_dir)), start=1):
                #     for idx, detection in objects.items():
                #         line = detection.data % (frame_idx, idx)
                #         fp.write(f"{line}\n")
