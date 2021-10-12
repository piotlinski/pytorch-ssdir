from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
from scipy.spatial.distance import cdist

from experiments.detect import Detection


class Tracker:
    def __init__(self, max_distance: float = 0.2, max_lost: int = 40):
        self.objects: Dict[int, Detection] = {}
        self._last_id = 0
        self.lost_counts: Dict[int, int] = defaultdict(int)
        self.max_distance = max_distance
        self.max_lost = max_lost

    def track(
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
                        [detection.centroid for detection in self.objects.values()]
                    ),
                    np.array([detection.centroid for detection in frame_detections]),
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
