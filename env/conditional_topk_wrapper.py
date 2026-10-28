import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv


class ConditionalTopKWrapper(gym.Wrapper):
    """
    Turn one K-subset decision into K conditional single-candidate decisions.

    The wrapped environment is only advanced after enough unique candidates have
    been selected for the current period. Intermediate steps expose the partial
    selected set as an extra observation row.
    """

    def __init__(self, env, K, invalid_penalty=0.0):
        super().__init__(env)
        self.K = int(K)
        self.invalid_penalty = float(invalid_penalty)
        self.action_space = env.action_space
        low = np.concatenate(
            [
                np.asarray(env.observation_space.low, dtype=np.float32),
                np.zeros((1, env.observation_space.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        high = np.concatenate(
            [
                np.asarray(env.observation_space.high, dtype=np.float32),
                np.ones((1, env.observation_space.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space.conditional_topk = True
        self.df = self._expanded_df()
        self.base_state = None
        self.selected_indices = []
        self.selected_mask = np.zeros(self.action_space.n, dtype=np.float32)

    def reset(self, **kwargs):
        self.base_state = self.env.reset(**kwargs)
        self.selected_indices = []
        self.selected_mask = np.zeros(self.action_space.n, dtype=np.float32)
        return self._augment_state()

    def step(self, action):
        idx = self._to_action_index(action)
        legal = self._legal_indices()
        target = min(self.K, len(legal))

        if target == 0:
            self.base_state, reward, done, info = self.env.step(np.array([], dtype=int))
            return self._reset_after_base_step(), reward, done, info

        if idx not in legal or idx in self.selected_indices:
            info = {
                "partial_selection": True,
                "invalid_action": True,
                "selected_indices": list(self.selected_indices),
            }
            return self._augment_state(), self.invalid_penalty, False, info

        self.selected_indices.append(idx)
        self.selected_mask[idx] = 1.0

        if len(self.selected_indices) < target:
            info = {
                "partial_selection": True,
                "invalid_action": False,
                "selected_indices": list(self.selected_indices),
            }
            return self._augment_state(), 0.0, False, info

        selected = np.array(self.selected_indices, dtype=int)
        self.base_state, reward, done, info = self.env.step(selected)
        info = dict(info)
        info["partial_selection"] = False
        info["invalid_action"] = False
        info["selected_indices"] = selected.tolist()
        return self._reset_after_base_step(), reward, done, info

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        if hasattr(self.env, "_seed"):
            return self.env._seed(seed)
        return None

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        state = e.reset()
        return e, state

    def save_asset_memory(self):
        return self.env.save_asset_memory()

    def save_truth_asset_memory(self):
        return self.env.save_truth_asset_memory()

    def save_tracking_error_memory(self):
        return self.env.save_tracking_error_memory()

    def save_daily_return_memory(self):
        return self.env.save_daily_return_memory()

    def save_action_memory(self):
        return self.env.save_action_memory()

    def save_metric_memory(self):
        return self.env.save_metric_memory()

    def _reset_after_base_step(self):
        self.selected_indices = []
        self.selected_mask = np.zeros(self.action_space.n, dtype=np.float32)
        return self._augment_state()

    def _augment_state(self):
        if self.base_state is None:
            self.base_state = self.env.render()
        return np.vstack([self.base_state, self.selected_mask.reshape(1, -1)]).astype(np.float32)

    def _legal_indices(self):
        if self.base_state is None:
            return []
        legal_mask = (np.asarray(self.base_state) != 0).any(axis=0)
        return np.flatnonzero(legal_mask).astype(int).tolist()

    @staticmethod
    def _to_action_index(action):
        flat = np.asarray(action).reshape(-1)
        if len(flat) == 0:
            return -1
        return int(flat[0])

    def _expanded_df(self):
        if not hasattr(self.env, "df"):
            return None
        df = self.env.df
        try:
            period_count = len(df.index.unique())
        except Exception:
            period_count = len(df)
        return df.iloc[:0].copy().reindex(range(period_count * max(1, self.K)))
