import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm

from experiments.detect import Representer


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
        whats = []
        wheres = []
        depths = []
        presents = []
        centroids = []
        for data_dir in pbar:
            pbar.desc = data_dir.name
            output_path = data_dir / args.output_file
            data = {}
            with output_path.open("wb") as fp:
                for frame_idx, image in enumerate(get_images(data_dir), start=1):
                    data[frame_idx] = list(detector(image))
                    for detection in data[frame_idx]:
                        whats.append(detection.what)
                        wheres.append(detection.where)
                        depths.append(detection.depth)
                        presents.append(detection.present)
                        centroids.append(detection.centroid)
                pickle.dump(data, fp)
        whats = np.array(whats)
        wheres = np.array(wheres)
        depths = np.array(depths)
        presents = np.array(presents)
        centroids = np.array(centroids)
        print(
            {
                output_path.name: {
                    "what": {
                        "min": whats.min(),
                        "max": whats.max(),
                        "mean": whats.mean(),
                        "std": whats.std(),
                    },
                    "where": {
                        "min": wheres.min(),
                        "max": wheres.max(),
                        "mean": wheres.mean(),
                        "std": wheres.std(),
                    },
                    "depth": {
                        "min": depths.min(),
                        "max": depths.max(),
                        "mean": depths.mean(),
                        "std": depths.std(),
                    },
                    "present": {
                        "min": presents.min(),
                        "max": presents.max(),
                        "mean": presents.mean(),
                        "std": presents.std(),
                    },
                    "centroid": {
                        "min": centroids.min(),
                        "max": centroids.max(),
                        "mean": centroids.mean(),
                        "std": centroids.std(),
                    },
                }
            }
        )
