import os

import pandas as pd

import config
from env.conditional_topk_wrapper import ConditionalTopKWrapper
from env.env_redset import RedsetAdmissionEnv, load_redset
from models import DRLAgent


def _conditional_topk_enabled(value=None):
    if value is None:
        value = os.environ.get("MC2P_CONDITIONAL_TOPK", "off")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _redset_path(dataset):
    paths = {
        "redset": "datasets/redset/provisioned/sample_0.01.parquet",
        "redset_smoke": "datasets/redset/serverless/sample_0.001.parquet",
        "redset_serverless_full": "datasets/redset/serverless/full.parquet",
        "redset_provisioned_sample": "datasets/redset/provisioned/sample_0.01.parquet",
    }
    return paths.get(dataset, dataset)


def train_redset(dataset, K, strategy):
    print("==============Start Fetching Redset Data===========")
    data_path = _redset_path(dataset)
    max_records = os.environ.get("REDSET_MAX_RECORDS")
    max_records = int(max_records) if max_records else None
    records = load_redset(data_path, max_records=max_records)

    # Split complete arrival periods to preserve chronological evaluation.
    split_period = int(records["arrival_period"].quantile(0.7))
    train_records = records[records["arrival_period"] <= split_period].reset_index(drop=True)
    test_records = records[records["arrival_period"] > split_period].reset_index(drop=True)
    if len(test_records) == 0:
        test_records = train_records.copy()

    max_pending = int(os.environ.get("REDSET_MAX_PENDING", "64"))
    episode_length = int(os.environ.get("REDSET_EPISODE_LENGTH", "200"))
    resource_budget = float(os.environ.get("REDSET_RESOURCE_BUDGET", "1.0"))
    w_min = float(os.environ.get("REDSET_W_MIN", "0.01"))

    env_kwargs = {
        "max_pending_queries": max_pending,
        "min_pending_queries": int(os.environ.get("REDSET_MIN_PENDING", str(int(K)))),
        "episode_length": episode_length,
        "resource_budget": resource_budget,
        "w_min": w_min,
        "lambda_D": float(os.environ.get("REDSET_LAMBDA_D", "0.8")),
        "lambda_W": float(os.environ.get("REDSET_LAMBDA_W", "0.2")),
        "lambda_P95": float(os.environ.get("REDSET_LAMBDA_P95", "0.0")),
        "eta_cpu": float(os.environ.get("REDSET_ETA_CPU", "0.5")),
        "eta_io": float(os.environ.get("REDSET_ETA_IO", "0.5")),
        "cost_budget": float(os.environ.get("REDSET_COST_BUDGET", "1.0")),
        "reward_chi": float(os.environ.get("REDSET_REWARD_CHI", "0.8")),
    }

    print("Redset path: {}".format(data_path))
    print("Train queries: {}, Test queries: {}".format(len(train_records), len(test_records)))
    print("Max pending queries: {}, K: {}".format(max_pending, K))

    e_train_gym = RedsetAdmissionEnv(train_records, **env_kwargs)
    e_test_gym = RedsetAdmissionEnv(test_records, **env_kwargs)
    conditional_topk = _conditional_topk_enabled()
    if conditional_topk:
        e_train_gym = ConditionalTopKWrapper(e_train_gym, K)
        e_test_gym = ConditionalTopKWrapper(e_test_gym, K)
    agent = DRLAgent(env=e_train_gym)

    model_kwargs = dict(config.DQN_PARAMS)
    model_kwargs["learning_starts"] = int(os.environ.get("REDSET_LEARNING_STARTS", "100"))
    model_kwargs["buffer_size"] = int(os.environ.get("REDSET_BUFFER_SIZE", "20000"))
    model_kwargs["batch_size"] = int(os.environ.get("REDSET_BATCH_SIZE", "64"))
    if os.environ.get("REDSET_SEED") is not None:
        model_kwargs["seed"] = int(os.environ["REDSET_SEED"])
    total_timesteps = int(os.environ.get("REDSET_TIMESTEPS", "5000"))

    print("==============Model Training===========")
    selected_num = 1 if conditional_topk else K
    print("Conditional Top-K: {}".format("on" if conditional_topk else "off"))
    model_dqn = agent.get_model("dqn", selected_num, strategy, model_kwargs=model_kwargs)
    model_save_path = "./" + config.TRAINED_MODEL_DIR + "/" + dataset + "/"
    os.makedirs(model_save_path, exist_ok=True)
    trained_dqn = agent.train_model(
        model=model_dqn,
        tb_log_name="dqn_redset",
        total_timesteps=total_timesteps,
        eval_env=e_test_gym,
        model_save_path=model_save_path,
    )

    print("==============Start Redset Testing===========")
    df_reward, df_objective, df_waiting = DRLAgent.DRL_prediction(
        model=trained_dqn,
        environment=e_test_gym,
    )
    os.makedirs("./" + config.RESULTS_DIR, exist_ok=True)
    df_reward.to_csv("./" + config.RESULTS_DIR + "/df_reward_" + dataset + ".csv")
    df_objective.to_csv("./" + config.RESULTS_DIR + "/df_objective_" + dataset + ".csv")
    df_waiting.to_csv("./" + config.RESULTS_DIR + "/df_waiting_" + dataset + ".csv")
    metric_cols = [
        "date",
        "gap",
        "wait",
        "cost",
        "completion_ratio",
        "throughput",
        "p95_waiting_time",
        "resource_utilization",
    ]
    if set(metric_cols).issubset(df_reward.columns):
        df_reward[metric_cols].to_csv("./" + config.RESULTS_DIR + "/df_metrics_" + dataset + ".csv")
    print("==============Redset Results===========")
    print(pd.concat(
        [
            df_reward.tail().rename(columns={"daily_return": "reward"}),
            df_objective.tail()["daily_return"].rename("objective"),
            df_waiting.tail()["daily_return"].rename("mean_waiting_time"),
        ],
        axis=1,
    ))
