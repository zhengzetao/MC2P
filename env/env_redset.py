import warnings

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv


FEATURE_COLUMNS = [
    "normalized_cpu_time",
    "normalized_scanned_bytes",
    "normalized_join_count",
    "normalized_agg_count",
    "normalized_sort_count",
    "remaining_demand",
    "waiting_time",
]


def _find_column(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def _log_normalize(series):
    values = np.log1p(pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0))
    high = values.quantile(0.99)
    if not np.isfinite(high) or high <= 0:
        high = values.max()
    if not np.isfinite(high) or high <= 0:
        high = 1.0
    return (values / high).clip(0.0, 1.0)


def load_redset(
    data,
    *,
    max_records=None,
    query_type="select",
    time_window_size="5min",
):
    """
    Load Redset query trace records and map real trace fields to MC2P state fields.

    Redset provides query metadata such as timestamps, execution duration,
    scanned bytes, and operator counts. The normalized fields, initial
    remaining_demand, beta_i, and CPU/I/O intensities below are derived values
    used by the lightweight simulator, not raw Redset measurements.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.read_parquet(data)

    columns = set(df.columns)
    timestamp_col = _find_column(columns, ["arrival_timestamp", "timestamp", "query_time"])
    cpu_col = _find_column(columns, ["cpu_time", "cpu_time_ms", "execution_duration_ms"])
    scanned_col = _find_column(columns, ["scanned_bytes", "bytes_scanned", "mbytes_scanned"])
    join_col = _find_column(columns, ["join_count", "num_joins"])
    agg_col = _find_column(columns, ["agg_count", "num_aggregations"])
    sort_col = _find_column(columns, ["sort_count", "num_sorts"])
    query_id_col = _find_column(columns, ["query_id"])

    required = {
        "timestamp": timestamp_col,
        "cpu/execution time": cpu_col,
        "scanned bytes": scanned_col,
    }
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError("Redset file is missing required fields: {}".format(", ".join(missing)))

    if sort_col is None:
        warnings.warn("Redset sort/operator count field not found; using sort_count=0.")

    if "query_type" in df.columns and query_type is not None:
        df = df[df["query_type"].astype(str).str.lower() == str(query_type).lower()]
    if "was_aborted" in df.columns:
        df = df[pd.to_numeric(df["was_aborted"], errors="coerce").fillna(0).astype(int) == 0]

    df = df.sort_values(timestamp_col).reset_index(drop=True)
    if max_records is not None:
        df = df.head(int(max_records)).copy()

    records = pd.DataFrame()
    records["query_id"] = df[query_id_col].values if query_id_col else np.arange(len(df))
    records["arrival_time"] = pd.to_datetime(df[timestamp_col], errors="coerce")
    records = records.dropna(subset=["arrival_time"]).reset_index(drop=True)
    df = df.loc[records.index].reset_index(drop=True)

    records["cpu_time"] = pd.to_numeric(df[cpu_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    records["scanned_bytes"] = pd.to_numeric(df[scanned_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    records["join_count"] = pd.to_numeric(df[join_col], errors="coerce").fillna(0.0) if join_col else 0.0
    records["agg_count"] = pd.to_numeric(df[agg_col], errors="coerce").fillna(0.0) if agg_col else 0.0
    records["sort_count"] = pd.to_numeric(df[sort_col], errors="coerce").fillna(0.0) if sort_col else 0.0

    records["normalized_cpu_time"] = _log_normalize(records["cpu_time"])
    records["normalized_scanned_bytes"] = _log_normalize(records["scanned_bytes"])
    records["normalized_join_count"] = _log_normalize(records["join_count"])
    records["normalized_agg_count"] = _log_normalize(records["agg_count"])
    records["normalized_sort_count"] = _log_normalize(records["sort_count"])
    records["remaining_demand_init"] = records["normalized_cpu_time"].clip(lower=1e-4)
    records["cpu_intensity"] = records["normalized_cpu_time"].clip(lower=1e-4)
    records["io_intensity"] = records["normalized_scanned_bytes"].clip(lower=1e-4)
    records["beta"] = (
        0.2
        + 0.5 * records["normalized_cpu_time"]
        + 0.3 * records["normalized_scanned_bytes"]
    )

    start = records["arrival_time"].min().floor(time_window_size)
    windows = records["arrival_time"].dt.floor(time_window_size)
    records["arrival_period"] = ((windows - start) / pd.Timedelta(time_window_size)).astype(int)

    ordered_cols = [
        "query_id",
        "arrival_time",
        "arrival_period",
        "cpu_time",
        "scanned_bytes",
        "join_count",
        "agg_count",
        "sort_count",
        "normalized_cpu_time",
        "normalized_scanned_bytes",
        "normalized_join_count",
        "normalized_agg_count",
        "normalized_sort_count",
        "remaining_demand_init",
        "cpu_intensity",
        "io_intensity",
        "beta",
    ]
    return records[ordered_cols].reset_index(drop=True)


class RedsetAdmissionEnv(gym.Env):
    """Multi-query admission and resource-allocation environment from Redset traces."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        records,
        max_pending_queries=64,
        min_pending_queries=5,
        episode_length=200,
        resource_budget=1.0,
        w_min=0.01,
        lambda_D=0.8,
        lambda_W=0.2,
        lambda_P95=0.0,
        eta_cpu=0.5,
        eta_io=0.5,
        cost_budget=1.0,
        reward_chi=0.8,
        reward_mode="cost_aware",
        epsilon_done=1e-3,
        day=0,
    ):
        self.records = records.sort_values(["arrival_period", "arrival_time"]).reset_index(drop=True)
        self.max_pending_queries = int(max_pending_queries)
        self.min_pending_queries = int(min_pending_queries)
        self.episode_length = int(episode_length)
        self.resource_budget = float(resource_budget)
        self.w_min = float(w_min)
        self.lambda_D = float(lambda_D)
        self.lambda_W = float(lambda_W)
        self.lambda_P95 = float(lambda_P95)
        self.eta_cpu = float(eta_cpu)
        self.eta_io = float(eta_io)
        self.cost_budget = float(cost_budget)
        self.reward_chi = float(reward_chi)
        self.reward_mode = reward_mode
        self.epsilon_done = float(epsilon_done)
        self.day = int(day)

        self.feature_dim = len(FEATURE_COLUMNS)
        self.action_space = spaces.Discrete(self.max_pending_queries)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self.feature_dim, self.max_pending_queries),
            dtype=np.float32,
        )
        self.df = pd.DataFrame(index=range(self.episode_length))
        self._initial_day = int(day)
        self.reset()

    def reset(self):
        self.day = self._initial_day
        self.current_period = 0
        self.cursor = 0
        self.pending = pd.DataFrame()
        self.remaining_demand = np.zeros(0, dtype=np.float32)
        self.waiting_time = np.zeros(0, dtype=np.float32)
        self.reward = 0.0
        self.terminal = False
        self.completed_total = 0
        self.last_allocation = np.zeros(self.max_pending_queries, dtype=np.float32)

        self.date_memory = [self.current_period]
        self.reward_memory = [0.0]
        self.objective_memory = [0.0]
        self.mean_waiting_memory = [0.0]
        self.action_memory = [range(0)]
        self.info_memory = []
        self.metric_memory = [
            {
                "date": self.current_period,
                "gap": 0.0,
                "wait": 0.0,
                "cost": 0.0,
                "completion_ratio": 0.0,
                "throughput": 0.0,
                "p95_waiting_time": 0.0,
                "resource_utilization": 0.0,
            }
        ]

        self._add_arrivals()
        self._fill_min_pending()
        return self._get_state()

    def step(self, actions):
        selected = self._valid_actions(actions)
        before_remaining = self.remaining_demand.copy()
        weights = self.allocate_resources(selected)
        cost = self._contention_cost(selected, weights[selected])
        cpu_pressure, io_pressure = self._aggregate_pressure(selected, weights[selected])

        progress = np.zeros(len(self.pending), dtype=np.float32)
        for idx in selected:
            progress[idx] = self._compute_progress_for_query(idx, weights[idx], selected, weights[selected])

        self.remaining_demand = np.maximum(0.0, self.remaining_demand - progress)
        selected_set = set(selected.tolist())
        for idx in range(len(self.waiting_time)):
            if idx not in selected_set:
                self.waiting_time[idx] += 1.0

        if len(selected) > 0:
            unchanged = [i for i in range(len(before_remaining)) if i not in selected_set]
            if unchanged:
                assert np.allclose(
                    self.remaining_demand[unchanged],
                    before_remaining[unchanged],
                )

        completed_mask = self.remaining_demand <= self.epsilon_done
        completed_now = int(completed_mask.sum())
        self.completed_total += completed_now
        if len(self.pending) > 0:
            self.pending = self.pending.loc[~completed_mask].reset_index(drop=True)
            self.remaining_demand = self.remaining_demand[~completed_mask]
            self.waiting_time = self.waiting_time[~completed_mask]

        self.current_period += 1
        self.day += 1
        self._add_arrivals()
        self._fill_min_pending()

        objective, mean_remaining, mean_waiting = self._objective()
        self.reward = self._reward(objective, cost)
        utilization = float(weights.sum())
        arrived = max(1, self.cursor)
        completion_ratio = float(self.completed_total / arrived)
        throughput = float(completed_now)
        p95_waiting = (
            float(np.percentile(self.waiting_time, 95))
            if len(self.waiting_time) > 0
            else 0.0
        )
        self.terminal = (
            self.day >= self.episode_length
            or (self.cursor >= len(self.records) and len(self.pending) == 0)
        )

        info = {
            "objective": objective,
            "mean_remaining_demand": mean_remaining,
            "mean_waiting_time": mean_waiting,
            "num_completed_queries": completed_now,
            "num_pending_queries": int(len(self.pending)),
            "resource_utilization": utilization,
            "mean_cpu_pressure": cpu_pressure,
            "mean_io_pressure": io_pressure,
            "decision_cost": cost,
            "gap": mean_remaining,
            "wait": mean_waiting,
            "cost": cost,
            "completion_ratio": completion_ratio,
            "throughput": throughput,
            "p95_waiting_time": p95_waiting,
            "data_terminal": bool(self.terminal),
        }
        self.info_memory.append(info)
        self.metric_memory.append(
            {
                "date": self.current_period,
                "gap": mean_remaining,
                "wait": mean_waiting,
                "cost": cost,
                "completion_ratio": completion_ratio,
                "throughput": throughput,
                "p95_waiting_time": p95_waiting,
                "resource_utilization": utilization,
            }
        )
        self.last_allocation = weights
        self.date_memory.append(self.current_period)
        self.reward_memory.append(self.reward)
        self.objective_memory.append(objective)
        self.mean_waiting_memory.append(mean_waiting)
        self.action_memory.append(selected.tolist())

        assert np.all(self.remaining_demand >= -1e-8)
        if len(selected) > 0:
            assert weights.sum() <= self.resource_budget + 1e-6

        return self._get_state(), self.reward, self.terminal, info

    def allocate_resources(self, selected):
        weights = np.zeros(self.max_pending_queries, dtype=np.float32)
        selected = np.array([i for i in selected if i < len(self.pending)], dtype=int)
        if len(selected) == 0:
            return weights

        min_total = self.w_min * len(selected)
        if min_total > self.resource_budget:
            base = self.resource_budget / len(selected)
            weights[selected] = base
            return weights

        weights[selected] = self.w_min
        remaining_budget = self.resource_budget - min_total
        beta = self._values("beta", selected)
        demand = self.remaining_demand[selected]
        cpu = self._values("cpu_intensity", selected)
        io = self._values("io_intensity", selected)
        score = demand / (beta + 0.5 * cpu + 0.5 * io + 1e-8)
        score = np.maximum(score, 1e-8)
        weights[selected] += remaining_budget * score / score.sum()
        return weights

    def _compute_progress_for_query(self, idx, weight, selected, selected_weights):
        selected = np.array(selected, dtype=int)
        selected_weights = np.array(selected_weights, dtype=np.float32)
        if idx >= len(self.pending) or weight <= 0:
            return 0.0

        beta = float(self.pending.iloc[idx]["beta"])
        cpu_i = float(self.pending.iloc[idx]["cpu_intensity"])
        io_i = float(self.pending.iloc[idx]["io_intensity"])
        other_mask = selected != idx
        other = selected[other_mask]
        other_weights = selected_weights[other_mask]

        cpu_contention = 0.0
        io_contention = 0.0
        if len(other) > 0:
            cpu_contention = cpu_i * float(np.sum(other_weights * self._values("cpu_intensity", other)))
            io_contention = io_i * float(np.sum(other_weights * self._values("io_intensity", other)))

        denom = beta + weight + self.eta_cpu * cpu_contention + self.eta_io * io_contention
        return float(weight / max(denom, 1e-8))

    def _get_state(self):
        state = np.zeros((self.feature_dim, self.max_pending_queries), dtype=np.float32)
        n = min(len(self.pending), self.max_pending_queries)
        if n == 0:
            return state
        raw = self.pending.iloc[:n][
            [
                "normalized_cpu_time",
                "normalized_scanned_bytes",
                "normalized_join_count",
                "normalized_agg_count",
                "normalized_sort_count",
            ]
        ].to_numpy(dtype=np.float32)
        dynamic = np.column_stack(
            [
                self.remaining_demand[:n],
                np.minimum(self.waiting_time[:n] / max(1, self.episode_length), 1.0),
            ]
        ).astype(np.float32)
        state[:, :n] = np.concatenate([raw, dynamic], axis=1).T
        return state

    def _add_arrivals(self):
        available_slots = self.max_pending_queries - len(self.pending)
        if available_slots <= 0:
            return
        start = self.cursor
        while self.cursor < len(self.records) and available_slots > 0:
            if int(self.records.iloc[self.cursor]["arrival_period"]) > self.current_period:
                break
            self.cursor += 1
            available_slots -= 1
        if self.cursor == start:
            return
        new_records = self.records.iloc[start:self.cursor].copy()
        self.pending = pd.concat([self.pending, new_records], ignore_index=True)
        self.remaining_demand = np.concatenate(
            [self.remaining_demand, new_records["remaining_demand_init"].to_numpy(dtype=np.float32)]
        )
        self.waiting_time = np.concatenate([self.waiting_time, np.zeros(len(new_records), dtype=np.float32)])

    def _fill_min_pending(self):
        target = min(self.min_pending_queries, self.max_pending_queries)
        while len(self.pending) < target and self.cursor < len(self.records):
            next_period = int(self.records.iloc[self.cursor]["arrival_period"])
            if next_period > self.current_period:
                self.current_period = next_period
            before = self.cursor
            self._add_arrivals()
            if self.cursor == before:
                break

    def _valid_actions(self, actions):
        flat = np.array(actions).reshape(-1)
        valid = []
        for action in flat:
            idx = int(action)
            if 0 <= idx < len(self.pending) and idx not in valid:
                valid.append(idx)
        return np.array(valid, dtype=int)

    def _values(self, column, indices):
        if len(indices) == 0:
            return np.zeros(0, dtype=np.float32)
        return self.pending.iloc[np.array(indices, dtype=int)][column].to_numpy(dtype=np.float32)

    def _aggregate_pressure(self, selected, selected_weights):
        if len(selected) == 0:
            return 0.0, 0.0
        cpu = float(np.sum(selected_weights * self._values("cpu_intensity", selected)))
        io = float(np.sum(selected_weights * self._values("io_intensity", selected)))
        return cpu, io

    def _contention_cost(self, selected, selected_weights):
        cpu_pressure, io_pressure = self._aggregate_pressure(selected, selected_weights)
        return float(0.5 * cpu_pressure + 0.5 * io_pressure)

    def _objective(self):
        if len(self.pending) == 0:
            return 0.0, 0.0, 0.0
        mean_remaining = float(np.mean(self.remaining_demand))
        normalized_waiting = np.minimum(self.waiting_time / max(1, self.episode_length), 1.0)
        mean_waiting = float(np.mean(normalized_waiting))
        p95_waiting = float(np.percentile(normalized_waiting, 95))
        objective = (
            self.lambda_D * mean_remaining
            + self.lambda_W * mean_waiting
            + self.lambda_P95 * p95_waiting
        )
        return float(objective), mean_remaining, mean_waiting

    def _shape(self, value):
        clipped = float(np.clip(value, 0.0, 1.0))
        return float((1.0 - clipped ** self.reward_chi) ** (1.0 / self.reward_chi))

    def _reward(self, objective, cost):
        if self.reward_mode == "negative_objective":
            return -float(objective)
        objective_part = self._shape(objective)
        cost_part = self._shape(min(1.0, cost / max(self.cost_budget, 1e-8)))
        return 0.5 * (objective_part + cost_part)

    def render(self, mode="human"):
        return self._get_state()

    def save_asset_memory(self):
        metrics = self.save_metric_memory()
        rewards = pd.DataFrame({"date": self.date_memory, "daily_return": self.reward_memory})
        return rewards.merge(metrics, on="date", how="left")

    def save_truth_asset_memory(self):
        return pd.DataFrame({"date": self.date_memory, "daily_return": self.objective_memory})

    def save_tracking_error_memory(self):
        return pd.DataFrame({"date": self.date_memory, "daily_return": self.mean_waiting_memory})

    def save_daily_return_memory(self):
        return pd.DataFrame(self.reward_memory)

    def save_action_memory(self):
        df_date = pd.DataFrame(self.date_memory)
        df_date.columns = ["date"]
        df_actions = pd.DataFrame(self.action_memory)
        df_actions.index = df_date.date
        return df_actions

    def save_metric_memory(self):
        return pd.DataFrame(self.metric_memory)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def seed(self, seed=None):
        return self._seed(seed)

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        state = e.reset()
        return e, state
