import argparse
import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from experiments.detect import Detection, Representer
from experiments.get_representations import get_images


def save_detection_results(
    detections: Iterable[Iterable[Detection]], results_path: Path
):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as fp:
        idx = 1
        for frame_idx, frame_detections in enumerate(detections, start=1):
            for detection in frame_detections:
                line = detection.data % (frame_idx, idx)
                fp.write(f"{line}\n")
                idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Path to the dataset", type=Path)
    parser.add_argument(
        "--checkpoint", "-c", help="Path to SSDIR checkpoint", type=Path
    )
    parser.add_argument(
        "--conf_range",
        nargs=2,
        type=int,
        default=[5, 100],
        help="Range of confidence threshold values",
    )
    parser.add_argument(
        "--nms_range",
        nargs=2,
        type=int,
        default=[5, 100],
        help="Range of NMS threshold values",
    )
    parser.add_argument("--output_dir", "-o", help="Path for results", type=Path)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Perform evaluation instead of detecting.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.evaluate:
        print("dupa")
        for confidence_threshold in range(*args.conf_range, 5):
            confidence_threshold /= 100
            for nms_threshold in range(*args.nms_range, 5):
                nms_threshold /= 100
                detector = Representer(
                    checkpoint=args.checkpoint,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                    max_per_image=100,
                )

                with tqdm(list(sorted(args.data_dir.glob("*/")))) as pbar:
                    for data_dir in pbar:
                        pbar.desc = data_dir.name
                        detections = (detector(image) for image in get_images(data_dir))
                        results_path = (
                            Path(args.output_dir)
                            / "MOT15-train"
                            / f"SSD_{confidence_threshold}_{nms_threshold}"
                            / "data"
                            / f"{data_dir.name}.txt"
                        )
                        save_detection_results(
                            (detector(image) for image in get_images(data_dir)),
                            results_path,
                        )
    else:
        subprocess.run(
            [
                "python",
                "experiments/run_mot_challenge.py",
                "--BENCHMARK=MOT15",
                f"--TRACKERS_FOLDER={args.output_dir}",
                "--GT_FOLDER=gt",
                "--METRICS=HOTA",
                "--USE_PARALLEL=True",
            ]
        )
        results_path = Path(args.output_dir) / "MOT15-train"
        dfs = []
        for setting_path in results_path.glob("*/"):
            _, conf_thresh, nms_thresh = setting_path.name.split("_")
            results_file = setting_path / "pedestrian_summary.txt"
            with results_file.open("r") as fp:
                loaded = pd.read_csv(fp, sep=" ")
                loaded.loc[loaded.index[0], "name"] = setting_path.name
                loaded.loc[loaded.index[0], "conf_thresh"] = float(conf_thresh)
                loaded.loc[loaded.index[0], "nms_thresh"] = float(nms_thresh)
                dfs.append(loaded)
        df = pd.concat(dfs, ignore_index=True).set_index("name").sort_index()
        print("Displaying top 10 settings - DetA")
        print(df.sort_values("DetA", ascending=False).head(10))
