import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from env.env_redset import FEATURE_COLUMNS, RedsetAdmissionEnv, load_redset


STATIC_FEATURE_COLUMNS = FEATURE_COLUMNS[:5]


def _legal_indices(env):
    return np.arange(min(len(env.pending), env.max_pending_queries), dtype=int)


def _top_k(indices, scores, K, largest=True):
    if len(indices) == 0:
        return np.array([], dtype=int)
    order = np.argsort(scores)
    if largest:
        order = order[::-1]
    return indices[order[: min(int(K), len(indices))]]


def random_policy(env, K, rng):
    indices = _legal_indices(env)
    if len(indices) <= K:
        return indices
    return rng.choice(indices, size=int(K), replace=False)


def fcfs_policy(env, K, rng=None):
    indices = _legal_indices(env)
    if len(indices) == 0:
        return indices
    arrival = env.pending.iloc[indices]["arrival_time"].astype("int64").to_numpy()
    return _top_k(indices, arrival, K, largest=False)


def sjf_policy(env, K, rng=None):
    indices = _legal_indices(env)
    if len(indices) == 0:
        return indices
    cpu = env.pending.iloc[indices]["normalized_cpu_time"].to_numpy(dtype=float)
    return _top_k(indices, cpu, K, largest=False)


def srpt_policy(env, K, rng=None):
    indices = _legal_indices(env)
    if len(indices) == 0:
        return indices
    return _top_k(indices, env.remaining_demand[indices], K, largest=False)


def max_wait_first_policy(env, K, rng=None):
    indices = _legal_indices(env)
    if len(indices) == 0:
        return indices
    return _top_k(indices, env.waiting_time[indices], K, largest=True)


def greedy_progress_policy(env, K, rng=None):
    indices = _legal_indices(env)
    if len(indices) == 0:
        return indices
    equal_weight = min(env.resource_budget / max(1, min(K, len(indices))), env.resource_budget)
    scores = []
    for idx in indices:
        progress = equal_weight / (env.pending.iloc[idx]["beta"] + equal_weight)
        scores.append(progress)
    return _top_k(indices, np.array(scores), K, largest=True)


def greedy_cost_aware_policy(env, K, rng=None):
    indices = _legal_indices(env)
    if len(indices) == 0:
        return indices
    equal_weight = min(env.resource_budget / max(1, min(K, len(indices))), env.resource_budget)
    scores = []
    for idx in indices:
        row = env.pending.iloc[idx]
        progress = equal_weight / (row["beta"] + equal_weight)
        pressure = 0.5 * row["cpu_intensity"] + 0.5 * row["io_intensity"]
        scores.append(progress - 0.25 * pressure)
    return _top_k(indices, np.array(scores), K, largest=True)


def auto_wlm_inspired_policy(env, K, rng=None):
    """
    Predictor-based scheduling baseline inspired by Redshift Auto-WLM.

    This is not an implementation of Amazon Redshift Auto-WLM. It uses Redset
    query metadata to estimate short, low-pressure queries and admit the top-K.
    """
    indices = _legal_indices(env)
    if len(indices) == 0:
        return indices
    rows = env.pending.iloc[indices]
    predicted_need = (
        0.45 * rows["normalized_cpu_time"].to_numpy(dtype=float)
        + 0.30 * rows["normalized_scanned_bytes"].to_numpy(dtype=float)
        + 0.15 * rows["normalized_join_count"].to_numpy(dtype=float)
        + 0.10 * rows["normalized_agg_count"].to_numpy(dtype=float)
    )
    wait_boost = np.minimum(env.waiting_time[indices] / max(1, env.episode_length), 1.0)
    scores = wait_boost - predicted_need
    return _top_k(indices, scores, K, largest=True)


class MLPTopKScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _candidate_features(env, indices, feature_mode="dynamic"):
    state = env._get_state().T
    features = state[np.asarray(indices, dtype=int)]
    if feature_mode == "static":
        return features[:, :len(STATIC_FEATURE_COLUMNS)]
    return features


def _one_step_priority_labels(env, indices, K):
    if len(indices) == 0:
        return np.zeros(0, dtype=np.float32)
    equal_weight = min(env.resource_budget / max(1, min(K, len(indices))), env.resource_budget)
    labels = []
    for idx in indices:
        row = env.pending.iloc[idx]
        beta = float(row["beta"])
        cpu = float(row["cpu_intensity"])
        io = float(row["io_intensity"])
        demand = float(env.remaining_demand[idx])
        wait = float(min(env.waiting_time[idx] / max(1, env.episode_length), 1.0))
        progress = equal_weight / max(beta + equal_weight + 0.25 * cpu + 0.25 * io, 1e-8)
        labels.append(min(demand, progress) + 0.2 * wait)
    return np.asarray(labels, dtype=np.float32)


def train_mlp_topk_policy(records, K=5, seed=0, feature_mode=None, **env_kwargs):
    """
    Supervised neural scoring baseline for Redset.

    It learns a unary candidate priority from the same Redset candidate features
    used by RA-DQN. Labels are simulator-derived one-step priorities under an
    equal-allocation assumption; no raw Redset field is treated as a reward.
    """
    torch.manual_seed(int(seed))
    rng = np.random.RandomState(seed)
    if feature_mode is None:
        feature_mode = os.environ.get("MLP_TOPK_FEATURE_MODE", "dynamic")
    env = RedsetAdmissionEnv(records, **env_kwargs)
    env.reset()
    xs = []
    ys = []
    horizon = min(int(env_kwargs.get("episode_length", env.episode_length)), 300)
    for _ in range(horizon):
        indices = _legal_indices(env)
        if len(indices) > 0:
            xs.append(_candidate_features(env, indices, feature_mode=feature_mode))
            ys.append(_one_step_priority_labels(env, indices, K))
            action = greedy_cost_aware_policy(env, K, rng)
        else:
            action = indices
        _, _, done, _ = env.step(action)
        if done:
            break

    if not xs:
        input_dim = len(STATIC_FEATURE_COLUMNS) if feature_mode == "static" else len(FEATURE_COLUMNS)
        model = MLPTopKScorer(input_dim)
        model.eval()
        return model

    x = torch.as_tensor(np.vstack(xs), dtype=torch.float32)
    y = torch.as_tensor(np.concatenate(ys), dtype=torch.float32)
    model = MLPTopKScorer(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(200):
        perm = torch.randperm(x.shape[0])
        for start in range(0, x.shape[0], 256):
            batch = perm[start:start + 256]
            pred = model(x[batch])
            loss = loss_fn(pred, y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def make_mlp_topk_policy(records, K=5, seed=0, feature_mode=None, **env_kwargs):
    if feature_mode is None:
        feature_mode = os.environ.get("MLP_TOPK_FEATURE_MODE", "dynamic")
    model = train_mlp_topk_policy(records, K=K, seed=seed, feature_mode=feature_mode, **env_kwargs)

    def policy(env, K, rng=None):
        indices = _legal_indices(env)
        if len(indices) == 0:
            return indices
        features = torch.as_tensor(_candidate_features(env, indices, feature_mode=feature_mode), dtype=torch.float32)
        with torch.no_grad():
            scores = model(features).cpu().numpy()
        return _top_k(indices, scores, K, largest=True)

    return policy


def mlp_topk_policy(env, K, rng=None):
    raise RuntimeError("mlp_topk requires a trained policy; call run_baseline('mlp_topk', ...).")


BASELINE_POLICIES = {
    "random": random_policy,
    "fcfs": fcfs_policy,
    "sjf": sjf_policy,
    "srpt": srpt_policy,
    "max_wait_first": max_wait_first_policy,
    "greedy_progress": greedy_progress_policy,
    "greedy_cost_aware": greedy_cost_aware_policy,
    "auto_wlm_inspired": auto_wlm_inspired_policy,
    "mlp_topk": mlp_topk_policy,
}


def run_baseline(method, records, K=5, seed=0, train_records=None, **env_kwargs):
    if method not in BASELINE_POLICIES:
        raise ValueError("Unknown Redset baseline: {}".format(method))

    rng = np.random.RandomState(seed)
    env = RedsetAdmissionEnv(records, **env_kwargs)
    env.reset()
    if method == "mlp_topk":
        policy = make_mlp_topk_policy(
            train_records if train_records is not None else records,
            K=K,
            seed=seed,
            **env_kwargs,
        )
    else:
        policy = BASELINE_POLICIES[method]

    for _ in range(env.episode_length):
        action = policy(env, K, rng)
        _, _, done, _ = env.step(action)
        if done:
            break

    metrics = env.save_metric_memory()
    metrics["method"] = method
    return metrics


def summarize_metrics(metrics):
    tail = metrics.tail(max(1, min(10, len(metrics))))
    return pd.Series(
        {
            "gap": tail["gap"].mean(),
            "wait": tail["wait"].mean(),
            "cost": tail["cost"].mean(),
            "completion_ratio": tail["completion_ratio"].max(),
            "throughput": metrics["throughput"].mean(),
            "p95_waiting_time": metrics["p95_waiting_time"].max(),
            "resource_utilization": metrics["resource_utilization"].mean(),
        }
    )


def _dataset_path(name):
    paths = {
        "redset": "datasets/redset/provisioned/sample_0.01.parquet",
        "redset_smoke": "datasets/redset/serverless/sample_0.001.parquet",
        "redset_serverless_full": "datasets/redset/serverless/full.parquet",
        "redset_provisioned_sample": "datasets/redset/provisioned/sample_0.01.parquet",
    }
    return paths.get(name, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="redset_smoke")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--methods", default=",".join(BASELINE_POLICIES.keys()))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--max-pending", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = _dataset_path(args.dataset)
    records = load_redset(path, max_records=args.max_records)
    methods = [name.strip() for name in args.methods.split(",") if name.strip()]

    os.makedirs("results", exist_ok=True)
    all_metrics = []
    summaries = []
    for method in methods:
        metrics = run_baseline(
            method,
            records,
            K=args.K,
            seed=args.seed,
            max_pending_queries=args.max_pending,
            min_pending_queries=args.K,
            episode_length=args.episode_length,
        )
        all_metrics.append(metrics)
        summary = summarize_metrics(metrics)
        summary["method"] = method
        summaries.append(summary)
        print("{}: {}".format(method, summary.to_dict()))

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    metrics_path = "results/redset_baseline_metrics_{}.csv".format(args.dataset)
    summary_path = "results/redset_baseline_summary_{}.csv".format(args.dataset)
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print("Saved {}".format(metrics_path))
    print("Saved {}".format(summary_path))


if __name__ == "__main__":
    main()
