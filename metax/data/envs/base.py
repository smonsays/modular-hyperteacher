"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc
from typing import Any, Tuple

import jax
from chex import Array, PRNGKey
from flax import struct

EnvironmentState = Any


@struct.dataclass
class EnvironmentInteraction:
    done: Array
    observation: Array
    reward: Array
    timestep: int
    info: dict = struct.field(pytree_node=False, default_factory=dict)


class Environment(abc.ABC):
    @abc.abstractproperty
    def num_actions(self) -> int:
        """ Number of possible actions."""

    @abc.abstractproperty
    def observation_shape(self):
        """The shape of the observation array"""

    @abc.abstractmethod
    def observe(self, env_state: EnvironmentState):
        """Returns the observation from the environment state."""

    @abc.abstractmethod
    def reset(self, rng: PRNGKey, goal: Array = None) -> Tuple[Any, EnvironmentInteraction]:
        """Resets the environment to an initial state."""

    @abc.abstractmethod
    def reset_goal(self, rng: PRNGKey, mode: str) -> Array:
        """Resets the environment goal."""

    def step(
        self, rng: PRNGKey, env_state: EnvironmentState, action: Array
    ) -> Tuple[EnvironmentState, EnvironmentInteraction]:
        """Run one timestep of the environment's dynamics. Returns the Transition and the Environment state."""

        # return self._step(rng, env_state, action)
        def empty_step(rng, state, action):
            """
            Only update time and give no reward.
            """
            new_timestep = state.timestep + 1
            new_state = state.replace(timestep=new_timestep)
            new_emission = EnvironmentInteraction(
                observation=self.observe(state),
                reward=0.0,
                done=state.done,
                timestep=new_timestep,
            )
            return new_state, new_emission

        # Only run env step if not already done
        return jax.lax.cond(
            env_state.done,
            empty_step,
            self._step,
            rng,
            env_state,
            action,
        )

    @abc.abstractmethod
    def _step(
        self, rng: PRNGKey, env_state: EnvironmentState, action: Array
    ) -> Tuple[EnvironmentState, EnvironmentInteraction]:
        """Run one timestep of the environment's dynamics. Returns the Transition and the Environment state."""
