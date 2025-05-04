from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """Initializes the observation and action space for the environment."""
        self.observation_space = gym.spaces.Discrete(2)  # two states (0 or 1)
        self.action_space = gym.spaces.Discrete(2)  # two actions (0 or 1)

        self.state = 0  # initial state


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Parameters
        ----------
        seed : int, optional
            Seed for environment reset (unused).
        options : dict, optional
            Additional reset options (unused).

        Returns
        -------
        state : int
            Initial state (always 0).
        info : dict
            An empty info dictionary.
        """
        self.current_steps = 0
        self.state = 0
        return self.state, {}
    

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]: 
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int
            Action to take (0 or 1).

        Returns
        -------
        next_state : int
            The resulting position.
        reward : float
            The reward at the new position.
        terminated : bool
            Whether the episode ended due to task success (always False here).
        truncated : bool
            Whether the episode ended due to reaching the time limit.
        info : dict
            An empty dictionary.
        """
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.current_steps += 1

        # deterministic transition
        # if action is 0 -> transition to state 0
        # if action is 1 -> transition to state 1
        previous_state = self.state
        self.state = action

        reward = self.get_reward_per_action()[previous_state, action]
        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, {}
    

    def get_reward_per_action(self) -> np.ndarray:
            """
            Return the reward function R[s, a] for each (state, action) pair.

            R[s, a] is the reward for taking action a in state s.

            Returns
            -------
            R : np.ndarray
                A (num_states, num_actions) array of rewards.
            """
            nS, nA = self.observation_space.n, self.action_space.n
            R = np.zeros((nS, nA), dtype=float)
            for s in range(nS):
                for a in range(nA):
                    R[s, a] = float(1 if s != a else 0)  # Reward is 1 if action leads to transition to the other state; else 0
            return R
    

    def get_transition_matrix(self) -> np.ndarray:
        """
        Construct a deterministic transition matrix T[s, a, s'].

        Returns
        -------
        T : np.ndarray
            A (num_states, num_actions, num_states) tensor where
            T[s, a, s'] = probability of transitioning to s' from s via a.
        """
        nS, nA = self.observation_space.n, self.action_space.n
        T = np.zeros((nS, nA, nS), dtype=float)
        for s in range(nS):
            for a in range(nA):
                s_next = a  # action is 0 => next state is 0; action is 1 => next state is 1
            T[s, a, s_next] = float(1)
        return T



class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        super().__init__(env)
        assert 0.0 <= noise <= 1.0, "noise must be in [0,1]"
        self.noise = noise
        self.rng = np.random.default_rng(seed)

        self.observation_space = env.observation_space
        self.action_space = env.action_space


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Reset the base environment and return a noisy observation.

        Parameters
        ----------
        seed : int or None, optional
            Seed for the reset, by default None.
        options : dict or None, optional
            Additional reset options, by default None.

        Returns
        -------
        obs : int
            The (possibly noisy) initial observation.
        info : dict
            Additional info returned by the environment.
        """
        true_obs, info = self.env.reset(seed=seed, options=options)
        return self._noisy_obs(true_obs), info
    

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment and return a noisy observation.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        obs : int
            The (possibly noisy) resulting observation.
        reward : float
            The reward received.
        terminated : bool
            Whether the episode terminated.
        truncated : bool
            Whether the episode was truncated due to time limit.
        info : dict
            Additional information from the base environment.
        """
        true_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._noisy_obs(true_obs), reward, terminated, truncated, info
    

    def _noisy_obs(self, true_obs: int) -> int:
        """
        Return a possibly noisy version of the true observation.

        With probability `noise`, replaces the true observation with
        a randomly selected incorrect state.

        Parameters
        ----------
        true_obs : int
            The true observation/state index.

        Returns
        -------
        obs : int
            A noisy (or true) observation.
        """
        if self.rng.random() < self.noise:
            return int(1 - true_obs)  # true_obs of 0 returns 1; true_obs of 1 returns 0
        else:
            return int(true_obs)
