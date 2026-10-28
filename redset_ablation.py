import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

from baseline_redset import run_baseline
from env.env_redset import load_redset


BASELINES = [
    "random",
    "fcfs",
    "sjf",
    "srpt",
    "greedy_cost_aware",
    "auto_wlm_inspired",
]


def _baseline_env_kwargs(args):
    return {
        "max_pending_queries": args.max_pending,
        "min_pending_queries": args.K,
        "episode_length": args.episode_length,
        "lambda_D": args.lambda_D,
        "lambda_W": args.lambda_W,
        "lambda_P95": args.lambda_P95,
        "eta_cpu": args.eta_cpu,
        "eta_io": args.eta_io,
    }


def summarize(metrics, window):
    tail = metrics.tail(max(1, min(int(window), len(metrics))))
    return {
        "gap": tail["gap"].mean(),
        "wait": tail["wait"].mean(),
        "cost": tail["cost"].mean(),
        "completion_ratio": tail["completion_ratio"].max(),
        "throughput": metrics["throughput"].mean(),
        "p95_waiting_time": metrics["p95_waiting_time"].max(),
        "resource_utilization": metrics["resource_utilization"].mean(),
    }


def copy_result(src, dst):
    if os.path.exists(src):
        shutil.copyfile(src, dst)


def run_dqn(args, strategy, seed, run_id, conditional_topk):
    env = os.environ.copy()
    env.update(
        {
            "TMPDIR": args.tmpdir,
            "REDSET_SEED": str(seed),
            "REDSET_TIMESTEPS": str(args.timesteps),
            "REDSET_LEARNING_STARTS": str(args.learning_starts),
            "REDSET_EPISODE_LENGTH": str(args.episode_length),
            "REDSET_MAX_PENDING": str(args.max_pending),
            "REDSET_MIN_PENDING": str(args.K),
            "REDSET_BUFFER_SIZE": str(args.buffer_size),
            "REDSET_BATCH_SIZE": str(args.batch_size),
            "REDSET_LAMBDA_D": str(args.lambda_D),
            "REDSET_LAMBDA_W": str(args.lambda_W),
            "REDSET_LAMBDA_P95": str(args.lambda_P95),
            "REDSET_ETA_CPU": str(args.eta_cpu),
            "REDSET_ETA_IO": str(args.eta_io),
        }
    )
    if args.max_records is not None:
        env["REDSET_MAX_RECORDS"] = str(args.max_records)
    env["MC2P_CONDITIONAL_TOPK"] = conditional_topk

    cmd = [
        sys.executable,
        "main.py",
        "--dataset",
        args.dataset,
        "--K",
        str(args.K),
        "--strategy",
        strategy,
        "--conditional_topk",
        conditional_topk,
    ]
    print("RUN", " ".join(cmd), "seed", seed, flush=True)
    subprocess.run(cmd, check=True, env=env)

    decoding = "conditional" if conditional_topk == "on" else "unary"
    prefix = "{}_{}_{}_seed{}".format(run_id, strategy, decoding, seed)
    metrics_path = "results/df_metrics_{}.csv".format(args.dataset)
    reward_path = "results/df_reward_{}.csv".format(args.dataset)
    copy_result(metrics_path, "results/{}_metrics.csv".format(prefix))
    copy_result(reward_path, "results/{}_reward.csv".format(prefix))
    metrics = pd.read_csv(metrics_path)
    row = summarize(metrics, args.metric_window)
    row.update({"method": "RA-DQN-{}-{}".format(strategy, decoding), "seed": seed})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", default="redset")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--strategies", default="concat,interactive")
    parser.add_argument("--conditional-modes", default="off,on")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--learning-starts", type=int, default=100)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--max-pending", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric-window", type=int, default=50)
    parser.add_argument("--lambda-D", type=float, default=0.8)
    parser.add_argument("--lambda-W", type=float, default=0.2)
    parser.add_argument("--lambda-P95", type=float, default=0.0)
    parser.add_argument("--eta-cpu", type=float, default=0.5)
    parser.add_argument("--eta-io", type=float, default=0.5)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--tmpdir", default=tempfile.gettempdir())
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs(args.tmpdir, exist_ok=True)

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    strategies = [name.strip() for name in args.strategies.split(",") if name.strip()]
    conditional_modes = [name.strip().lower() for name in args.conditional_modes.split(",") if name.strip()]

    records = load_redset("datasets/redset/provisioned/sample_0.01.parquet", max_records=args.max_records)
    rows = []

    for seed in seeds:
        for method in BASELINES:
            print("BASELINE", method, "seed", seed, flush=True)
            metrics = run_baseline(
                method,
                records,
                K=args.K,
                seed=seed,
                **_baseline_env_kwargs(args),
            )
            metrics.to_csv(
                "results/{}_{}_seed{}_metrics.csv".format(args.run_id, method, seed),
                index=False,
            )
            row = summarize(metrics, args.metric_window)
            row.update({"method": method, "seed": seed})
            rows.append(row)

    for seed in seeds:
        for strategy in strategies:
            for conditional_topk in conditional_modes:
                rows.append(run_dqn(args, strategy, seed, args.run_id, conditional_topk))

    per_seed = pd.DataFrame(rows)
    per_seed_path = "results/{}_per_seed_summary.csv".format(args.run_id)
    per_seed.to_csv(per_seed_path, index=False)

    metrics = [
        "gap",
        "wait",
        "cost",
        "completion_ratio",
        "throughput",
        "p95_waiting_time",
        "resource_utilization",
    ]
    summary = per_seed.groupby("method")[metrics].agg(["mean", "std"])
    summary.columns = ["{}_{}".format(metric, stat) for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary_path = "results/{}_aggregate_summary.csv".format(args.run_id)
    summary.to_csv(summary_path, index=False)

    primary = summary[
        ["method", "gap_mean", "gap_std", "wait_mean", "wait_std", "cost_mean", "cost_std"]
    ]
    primary_path = "results/{}_three_metrics_summary.csv".format(args.run_id)
    primary.to_csv(primary_path, index=False)
    print(primary.to_string(index=False), flush=True)
    print("Saved", per_seed_path, summary_path, primary_path, flush=True)


if __name__ == "__main__":
    main()
