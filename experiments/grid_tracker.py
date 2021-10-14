import argparse
import pickle
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from experiments.track import R, Tracker, save_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Path to the dataset", type=Path)
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Path to directory where results should be stored",
        default="results/grid_track",
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
        "--max_distance_range",
        nargs=2,
        type=int,
        default=[5, 100],
        help="Range of max distance between objects values",
    )
    parser.add_argument(
        "--max_lost_range",
        nargs=2,
        type=int,
        default=[5, 100],
        help="Range of max frames before dropping lost detection",
    )
    parser.add_argument(
        "--metric_values",
        nargs="+",
        default=["cosine", "euclidean"],
        help="Metrics for cdist to calculate distance between centroids",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Perform evaluation instead of tracking.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_distance_range = list(range(*args.max_distance_range, 5))
    max_lost_range = list(range(*args.max_lost_range, 5))
    metric_values = list(args.metric_values)
    if not args.evaluate:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        max_distance_range = list(range(*args.max_distance_range, 5))
        max_lost_range = list(range(*args.max_lost_range, 5))
        metric_values = list(args.metric_values)
        with tqdm(
            total=len(max_distance_range) * len(max_lost_range) * len(metric_values)
        ) as top_pbar:
            for max_distance in max_distance_range:
                max_distance /= 100
                for max_lost in max_lost_range:
                    for metric in metric_values:
                        tracker_name = (
                            f"SSDIR_{max_distance}_{max_lost}_{metric}"
                            f"_{'-'.join(args.representations)}"
                        )
                        top_pbar.desc = tracker_name
                        with tqdm(list(sorted(args.data_dir.glob("*/")))) as pbar:
                            for data_dir in pbar:
                                pbar.desc = data_dir.name
                                tracker = Tracker(
                                    max_distance=max_distance,
                                    max_lost=max_lost,
                                    metric=metric,
                                    centroids=[
                                        R[selected.upper()]
                                        for selected in args.representations
                                    ],
                                )
                                results_path = (
                                    Path(args.output_dir)
                                    / "MOT15-train"
                                    / tracker_name
                                    / "data"
                                    / f"{data_dir.name}.txt"
                                )
                                latents_path = data_dir / args.input_file
                                with latents_path.open("rb") as fp:
                                    latents = pickle.load(fp)
                                save_results(tracker(latents.values()), results_path)
                        top_pbar.update()
    else:
        subprocess.run(
            [
                "python",
                "experiments/run_mot_challenge.py",
                "--BENCHMARK=MOT15",
                f"--TRACKERS_FOLDER={args.output_dir}",
                "--GT_FOLDER=results/gt",
                "--METRICS=HOTA",
                "--USE_PARALLEL=True",
                "--NUM_PARALLEL_CORES=40",
            ]
        )
        results_path = Path(args.output_dir) / "MOT15-train"
        dfs = []
        for setting_path in results_path.glob("*/"):
            _, max_distance, max_lost, metric, representation = setting_path.name.split(
                "_"
            )
            results_file = setting_path / "pedestrian_summary.txt"
            with results_file.open("r") as fp:
                loaded = pd.read_csv(fp, sep=" ")
                loaded.loc[loaded.index[0], "name"] = setting_path.name
                loaded.loc[loaded.index[0], "max_distance"] = float(max_distance)
                loaded.loc[loaded.index[0], "max_lost"] = int(max_lost)
                loaded.loc[loaded.index[0], "metric"] = metric
                loaded.loc[loaded.index[0], "representation"] = representation
                dfs.append(loaded)
        df = pd.concat(dfs, ignore_index=True).set_index("name").sort_index()
        print("Displaying top 10 settings - HOTA")
        print(df.sort_values("HOTA", ascending=False).head(10))
