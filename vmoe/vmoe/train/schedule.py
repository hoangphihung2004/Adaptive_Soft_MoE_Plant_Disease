# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Learning rate schedules."""
import math

import jax
import optax


def create_learning_rate_schedule(*,
                                  schedule: str,
                                  total_steps: int,
                                  **kwargs) -> optax.Schedule:
  """Creates a optax learning rate schedule."""
  if 'base_learning_rate' in kwargs:
    kwargs['peak_value'] = kwargs.pop('base_learning_rate')
  if 'warmup_epochs' in kwargs:
    warmup_epochs = kwargs.pop('warmup_epochs')
    kwargs['warmup_steps'] = int((warmup_epochs / 100.0) * total_steps)
  if 'warmup_steps' not in kwargs:
    kwargs['warmup_steps'] = max(1, int(0.05 * total_steps))

  if schedule == 'constant':
    val = kwargs.get('peak_value', kwargs.get('value', 1e-3))
    return optax.constant_schedule(val)

  kwargs['decay_steps'] = kwargs.pop('decay_steps', total_steps)

  if schedule in ('warmup_cosine', 'warmup_cosine_decay'):
    peak_val = kwargs.pop('peak_value', 5e-6)
    warmup_st = kwargs.pop('warmup_steps', 500)
    decay_st = kwargs.pop('decay_steps', total_steps)
    end_val = kwargs.pop('end_value', 1e-7)
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_val,
        warmup_steps=warmup_st,
        decay_steps=decay_st,
        end_value=end_val
    )
  if schedule == 'warmup_linear_decay':
    return warmup_polynomial_decay_schedule(power=1.0, **kwargs)
  if schedule == 'warmup_polynomial_decay':
    return warmup_polynomial_decay_schedule(**kwargs)
  if schedule == 'big_vision_rsqrt':
    return big_vision_rsqrt_schedule(**kwargs)
  # Unknown learning rate schedule.
  raise ValueError(f'Unknown learning rate schedule: {schedule!r}')


def warmup_polynomial_decay_schedule(peak_value: float, end_value: float,
                                     power: float, warmup_steps: int,
                                     decay_steps: int) -> optax.Schedule:
  """Linear warmup followed by polynomial decay."""
  return optax.join_schedules([
      optax.linear_schedule(
          init_value=0.0, end_value=peak_value, transition_steps=warmup_steps),
      optax.polynomial_schedule(
          init_value=peak_value,
          end_value=end_value,
          power=power,
          transition_steps=decay_steps - warmup_steps)
  ], [warmup_steps])


def big_vision_rsqrt_schedule(
    *,
    peak_value: float,
    decay_steps: int,
    timescale: int,
    warmup_steps: int = 0,
    cooldown_steps: int = 0) -> optax.Schedule:
  """Inverse Sqrt schedule inspired by the "Scaling Vision Transformers" paper.

  It performs a linear warmup, followed by a inverse sqrt decay, and finally a
  linear cooldown to 0. This is useful to continue training the model more steps
  than originally planned, by restarting from a checkpoint before the cooldown.
  See https://arxiv.org/abs/2106.04560 for additional details.

  Args:
    peak_value: Peak value for scalar to be annealed at end of warmup.
    decay_steps: Positive integer, the total length of the schedule.
    timescale: Positive integer, the count is divided by this number during the
      inverse sqrt phase. Larger values reduce the rate of the decay.
    warmup_steps: Positive integer, the length of the linear warmup.
    cooldown_steps: Positive integer, the length of the linear cooldown.

  Returns:
   A optax.Schedule object.
  """
  if decay_steps < warmup_steps + cooldown_steps:
    raise ValueError('decay_steps must be greater than or equal to '
                     f'warmup_steps + cooldown_steps, but got {decay_steps}, '
                     f'{warmup_steps} and {cooldown_steps}, respectively.')
  if timescale < 0:
    raise ValueError(f'timescale must be positive, but got {timescale}')

  # This is the value of the learning rate just before the cooldown.
  # Note: big_vision rsqrt actually applies the 1 / sqrt(...)  factor also
  # during cooldown, but that seems a bug and the differences are negligible.
  cooldown_peak_value = peak_value / math.sqrt(
      1. + (decay_steps - warmup_steps - cooldown_steps) / timescale)

  def _rsqrt(count):
    # Note: count starts from 0, optax.join_schedules subtracts warmup_steps.
    return peak_value * jax.lax.rsqrt(1. + count / timescale)

  def _cooldown(count):
    # Note: count starts from 0, optax.join_schedules subtracts
    # decay_steps - cooldown_steps.
    return cooldown_peak_value * (1 - count / cooldown_steps)

  schedules, boundaries = [], []
  # Warmup phase.
  if warmup_steps > 0:
    schedules.append(
        optax.linear_schedule(
            init_value=0.0, end_value=peak_value,
            transition_steps=warmup_steps))
    boundaries.append(warmup_steps)
  # Main schedule.
  schedules.append(_rsqrt)
  boundaries.append(decay_steps - cooldown_steps)
  # Cooldown phase.
  if cooldown_steps > 0:
    schedules.append(_cooldown)

  return optax.join_schedules(schedules, boundaries)
