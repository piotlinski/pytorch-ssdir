import argparse
import pickle
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from chardet import detect
from scipy.spatial.distance import cdist
from tqdm import tqdm

from experiments.detect import Detection


class R(Enum):
    CENTROID = "centroid"
    WHERE = "where"
    PRESENT = "present"
    DEPTH = "depth"
    WHAT = "what"


class Tracker:
    def __init__(
        self,
        max_distance: float = 0.2,
        max_lost: int = 40,
        metric: str = "cosine",
        centroids: Optional[List[R]] = None,
    ):
        self.objects: Dict[int, Detection] = {}
        self._last_id = 0
        self.lost_counts: Dict[int, int] = defaultdict(int)
        self.max_distance = max_distance
        self.max_lost = max_lost
        self.metric = metric
        self.centroids = centroids or [R.CENTROID]

    def __call__(
        self, detections: Iterable[Iterable[Detection]]
    ) -> Iterable[Dict[int, Detection]]:
        """Perform object tracking"""
        for frame_detections in detections:
            frame_detections = list(frame_detections)
            object_ids = list(self.objects.keys())
            if not self.objects:
                for detection in frame_detections:
                    self._register(detection)
            elif frame_detections:
                distances = cdist(
                    np.array(
                        [
                            self.get_centroid(detection)
                            for detection in self.objects.values()
                        ]
                    ),
                    np.array(
                        [self.get_centroid(detection) for detection in frame_detections]
                    ),
                    metric=self.metric,
                )
                rows = distances.min(axis=1).argsort()
                cols = distances.argmin(axis=1)[rows]

                used_rows = set()
                used_cols = set()

                for row, col in zip(rows, cols):
                    if (
                        row in used_rows
                        or col in used_cols
                        or distances[row, col] > self.max_distance
                    ):
                        continue

                    object_id = object_ids[row]
                    self.objects[object_id] = frame_detections[col]
                    self.lost_counts[object_id] = 0
                    used_rows.add(row)
                    used_cols.add(col)

                if distances.shape[0] >= distances.shape[1]:
                    for row in set(range(distances.shape[0])).difference(used_rows):
                        self._increase_lost_count(object_ids[row])
                else:
                    for col in set(range(distances.shape[1])).difference(used_cols):
                        self._register(frame_detections[col])
            else:
                for object_id in object_ids:
                    self._increase_lost_count(object_id)
            yield self.objects

    def _register(self, detection: Detection):
        """Register new detection"""
        self.objects[self._last_id] = detection
        self._last_id += 1

    def _deregister(self, object_id: int):
        """Deregister detection"""
        del self.objects[object_id]

    def _increase_lost_count(self, object_id: int):
        """Increase lost count for the object"""
        self.lost_counts[object_id] += 1
        if self.lost_counts[object_id] > self.max_lost:
            self._deregister(object_id)
            del self.lost_counts[object_id]

    def get_centroid(self, detection: Detection) -> np.ndarray:
        centroid = []
        for selected in self.centroids:
            centroid.append(getattr(detection, selected.value))
        return np.concatenate(centroid)


def save_results(objects: Iterable[Dict[int, Detection]], results_path: Path):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as fp:
        for frame_idx, frame_objects in enumerate(objects, start=1):
            for idx, detection in frame_objects.items():
                line = detection.data % (frame_idx, idx)
                fp.write(f"{line}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Path to the dataset", type=Path)
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Path to directory where results should be stored",
        default="results",
        type=Path,
    )
    parser.add_argument("-i", "--input_file", help="Pickle file name with latents")
    parser.add_argument(
        "-r",
        "--representations",
        nargs="+",
        help="Select representations used by tracker",
    )
    parser.add_argument(
        "--max_distance",
        help="Max distance between objects for tracker",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--max_lost",
        help="Max frames before dropping lost detection",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--metric",
        help="Metric for cdist to calculate distance between centroids",
        default="cosine",
    )

    args = parser.parse_args()
    output_dir = args.output_dir / "MOT15-train"

    with tqdm(list(sorted(args.data_dir.glob("*/")))) as pbar:
        for data_dir in pbar:
            tracker = Tracker(
                max_distance=args.max_distance,
                max_lost=args.max_lost,
                metric=args.metric,
                centroids=[R[selected.upper()] for selected in args.representations],
            )
            latents_path = data_dir / args.input_file
            with latents_path.open("rb") as fp:
                latents = pickle.load(fp)
            tracker_name = (
                f"SSDIR_{args.input_file.split('.')[0]}"
                f"_{'-'.join(args.representations)}_{args.metric}"
            )
            results_path = output_dir / tracker_name / "data" / f"{data_dir.name}.txt"
            save_results(tracker(latents.values()), results_path)
