# ddsp_phasor_core

MODULE NAME:
**ddsp_smoothing_core**

DESCRIPTION:
A collection of fully differentiable, GDSP-style smoothing primitives in pure JAX.
It provides **one-pole exponential smoothing**, **linear ramp smoothing**, and a **first-order allpass smoother**.
Each smoother is exposed with `*_init`, `*_tick`, and `*_process` functions, suitable for use in oscillators, filters, envelopes, and UI parameter smoothing.

INPUTS:

### One-pole smoother

* **x** : input control/sample to be smoothed
* **params.alpha** : smoothing coefficient in `[0,1]` (0 = no smoothing, 1 = heavy smoothing)

### Linear ramp smoother

* **x** : desired target value (can be time-varying)
* **params.duration_samples** : number of samples over which to ramp from current value to the new target

### Allpass smoother

* **x** : input control/sample to be smoothed via a first-order allpass
* **params.a** : allpass coefficient, clipped to `(-1, 1)` (stable if `|a| < 1`)

OUTPUTS:

For all smoothers:

* **y** : smoothed output sample

STATE VARIABLES:

### One-pole smoother

`state = (prev_y,)`

* **prev_y** : previous output sample `y[n−1]`

### Linear ramp smoother

`state = (current_value, target_value, increment, remaining_samples)`

* **current_value** : last ramped value
* **target_value** : current ramp target
* **increment** : per-sample increment to move from `current_value` to `target_value`
* **remaining_samples** : how many ramp steps remain (float, treated as non-negative)

### Allpass smoother

`state = (prev_x, prev_y)`

* **prev_x** : previous input sample `x[n−1]`
* **prev_y** : previous output sample `y[n−1]`

EQUATIONS / MATH:

### One-pole exponential smoothing

One-sample tick:

* `y[n] = y[n−1] + α (x[n] − y[n−1])`

State update:

* `prev_y[n+1] = y[n]`

### Linear ramp smoothing

At each tick, we want to track a ramp from `current_value` to `target_value` over `duration_samples` ticks.
A new ramp is triggered when:

* `remaining_samples <= 0`, or
* `x[n]` (new desired target) differs from `target_value` by more than a small threshold.

When a new ramp starts:

* `target_value_new = x[n]`
* `increment_new = (target_value_new − current_value) / duration_samples`
* `remaining_samples_new = duration_samples`

On each tick:

* `value_candidate = current_value + increment`
* `remaining_samples_next = max(remaining_samples − 1, 0)`
* `finished = (remaining_samples_next <= 1)`

Output:

* `y[n] = finished ? target_value : value_candidate` (implemented via `jnp.where`)

State update:

* `current_value[n+1] = y[n]`
* `target_value[n+1] = target_value_next`
* `increment[n+1] = increment_next`
* `remaining_samples[n+1] = remaining_samples_next`

All logic uses `jnp.where` and masks, so no Python branching is used inside jit.

### Allpass smoother

First-order allpass filter:

* `y[n] = a * x[n] + prev_x − a * prev_y`

State update:

* `prev_x[n+1] = x[n]`
* `prev_y[n+1] = y[n]`

Coefficient stability:

* `a_clip = clip(a, -0.999, 0.999)`

NOTES:

* All smoothers are **fully differentiable**, no `stop_gradient` calls.
* All per-sample control logic uses `jnp.where` and masks, no Python `if` inside `@jax.jit`.
* `process()` for each smoother is a `lax.scan` over `tick()`.
* No allocation of new arrays inside jit, no `jnp.arange` or `jnp.zeros` inside jit.
* All shapes are determined from inputs outside jit.
* Parameters can be scalar or per-sample arrays; they are broadcast before scan.
* The linear ramp smoother is robust to time-varying targets and durations and behaves like a **Gamma-style ramped parameter**.

---

## Full `ddsp_smoothing_core.py`

```python
"""
ddsp_smoothing_core.py

GammaJAX DDSP – Smoothing Core
------------------------------

This module implements several fully differentiable, GDSP-style smoothing
primitives in pure JAX:

    - One-pole exponential smoother
    - Linear ramp smoother (Gamma::Ramped style)
    - First-order allpass smoother

Each smoother follows the GDSP API:

    <name>_init(...)
    <name>_update_state(...)
    <name>_tick(x, state, params)
    <name>_process(xs, state, params)

Requirements:
    - Pure functional JAX, no side effects.
    - No classes, no dicts, no dataclasses.
    - State = tuple only (arrays, scalars).
    - No Python branching inside @jax.jit; use jnp.where / lax.cond.
    - No dynamic allocation inside jit; all shapes determined outside jit.
    - No jnp.arange / jnp.zeros inside jit.
    - Functional DSP: tick returns (y, new_state).
    - process() is a lax.scan wrapper around tick().
    - Everything differentiable and jit-safe.

Smoothers:

1. One-pole:
    y[n] = y[n-1] + alpha * (x[n] - y[n-1])

    State: (prev_y,)
    Params: (alpha,)

2. Linear ramp:
    State: (current_value, target_value, increment, remaining_samples)
    Params: (duration_samples,)

3. Allpass smoother:
    y[n] = a * x[n] + prev_x - a * prev_y

    State: (prev_x, prev_y)
    Params: (a,)
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax


# =============================================================================
# 1. One-pole exponential smoother
# =============================================================================

def onepole_smoother_init(
    initial_value: float,
    alpha: float,
    *,
    dtype=jnp.float32,
) -> Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray]]:
    """
    Initialize one-pole smoother.

    Args:
        initial_value : starting value y[0]
        alpha         : smoothing coefficient in [0,1]
                        0  => no smoothing (y=x)
                        1  => heavy smoothing / slow response
        dtype         : JAX dtype

    Returns:
        state  : (prev_y,)
        params : (alpha,)
    """
    prev_y = jnp.asarray(initial_value, dtype=dtype)
    alpha_arr = jnp.asarray(alpha, dtype=dtype)
    state = (prev_y,)
    params = (alpha_arr,)
    return state, params


def onepole_smoother_update_state(
    state: Tuple[jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray]:
    """
    Placeholder update_state for one-pole smoother.
    This smoother's state only changes during tick(), so we simply return state.
    """
    return state


@jax.jit
def onepole_smoother_tick(
    x: jnp.ndarray,
    state: Tuple[jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
    """
    One-pole smoother single tick.

    Args:
        x      : input sample
        state  : (prev_y,)
        params : (alpha,)

    Returns:
        y        : smoothed sample
        new_state: (prev_y_next,)
    """
    (prev_y,) = state
    (alpha,) = params

    prev_y = jnp.asarray(prev_y, dtype=x.dtype)
    alpha = jnp.asarray(alpha, dtype=x.dtype)
    alpha = jnp.clip(alpha, 0.0, 1.0)

    y = prev_y + alpha * (x - prev_y)
    new_state = (y,)
    return y, new_state


@jax.jit
def onepole_smoother_process(
    xs: jnp.ndarray,
    state: Tuple[jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
    """
    Process a buffer with one-pole smoother via lax.scan.

    Args:
        xs     : input buffer, shape (T,)
        state  : (prev_y,)
        params : (alpha,), where alpha may be scalar or shape (T,)

    Returns:
        ys         : smoothed buffer, shape (T,)
        final_state
    """
    (prev_y,) = state
    (alpha,) = params

    xs = jnp.asarray(xs)
    T = xs.shape[0]

    alpha = jnp.asarray(alpha, dtype=xs.dtype)
    alpha = jnp.broadcast_to(alpha, (T,))

    init_state = (prev_y,)

    def body(carry, xs_t):
        st = carry
        x_t, a_t = xs_t
        y_t, st_next = onepole_smoother_tick(x_t, st, (a_t,))
        return st_next, y_t

    final_state, ys = lax.scan(body, init_state, (xs, alpha))
    return ys, final_state


# =============================================================================
# 2. Linear ramp smoother
# =============================================================================

def ramp_smoother_init(
    initial_value: float,
    initial_target: float,
    duration_samples: float,
    *,
    dtype=jnp.float32,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
           Tuple[jnp.ndarray]]:
    """
    Initialize linear ramp smoother.

    The ramp smoother tracks a value from current_value to target_value over
    duration_samples ticks. A new ramp is triggered whenever the incoming
    target (x) changes significantly or when the previous ramp finishes.

    Args:
        initial_value    : starting output value
        initial_target   : initial target to ramp toward
        duration_samples : number of samples to reach the target
        dtype            : JAX dtype

    Returns:
        state  : (current_value, target_value, increment, remaining_samples)
        params : (duration_samples,)
    """
    cv = jnp.asarray(initial_value, dtype=dtype)
    tv = jnp.asarray(initial_target, dtype=dtype)
    dur = jnp.asarray(duration_samples, dtype=dtype)
    dur = jnp.maximum(dur, 1.0)

    inc = (tv - cv) / dur
    remaining = dur

    state = (cv, tv, inc, remaining)
    params = (dur,)
    return state, params


def ramp_smoother_update_state(
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Placeholder update_state for ramp smoother.
    Ramp smoothing is driven by tick() based on target inputs.
    """
    return state


@jax.jit
def ramp_smoother_tick(
    x: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Linear ramp smoother single tick.

    Inputs:
        x      : new desired target value at this sample
        state  : (current_value, target_value, increment, remaining_samples)
        params : (duration_samples,)

    Behavior:
        - If remaining_samples <= 0 or the target has changed by more than
          a small threshold, start a new ramp from current_value to x over
          duration_samples ticks.
        - Otherwise, continue existing ramp.

    All control flow uses jnp.where masks, no Python branching.
    """
    (current_value, target_value, increment, remaining) = state
    (duration_samples,) = params

    x = jnp.asarray(x, dtype=current_value.dtype)
    duration_samples = jnp.asarray(duration_samples, dtype=current_value.dtype)
    duration_samples = jnp.maximum(duration_samples, 1.0)

    # Condition for starting a new ramp
    diff = jnp.abs(x - target_value)
    changed = diff > 1e-12
    finished = remaining <= 0.5

    changed_mask = jnp.where(changed, 1.0, 0.0)
    finished_mask = jnp.where(finished, 1.0, 0.0)
    new_ramp_flag = jnp.maximum(changed_mask, finished_mask)

    # Candidate new increment/target if ramp is (re)started
    inc_candidate = (x - current_value) / duration_samples
    increment_next = jnp.where(new_ramp_flag > 0.5, inc_candidate, increment)
    target_next = jnp.where(new_ramp_flag > 0.5, x, target_value)
    remaining_candidate = jnp.where(new_ramp_flag > 0.5,
                                    duration_samples,
                                    jnp.maximum(remaining - 1.0, 0.0))

    # Advance value
    value_candidate = current_value + increment_next

    # Determine whether current ramp is finished after this step
    finished_after = remaining_candidate <= 1.0
    finished_after_mask = jnp.where(finished_after, 1.0, 0.0)

    # Output is either the ramped value or the target when finished
    y = jnp.where(finished_after_mask > 0.5, target_next, value_candidate)

    current_value_next = y
    remaining_next = jnp.where(finished_after_mask > 0.5, 0.0, remaining_candidate)

    new_state = (current_value_next, target_next, increment_next, remaining_next)
    return y, new_state


@jax.jit
def ramp_smoother_process(
    xs: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Process a buffer with the ramp smoother.

    Args:
        xs     : buffer of target values, shape (T,)
        state  : (current_value, target_value, increment, remaining_samples)
        params : (duration_samples,), scalar or shape (T,)

    Returns:
        ys         : ramped buffer, shape (T,)
        final_state
    """
    xs = jnp.asarray(xs)
    (duration_samples,) = params
    duration_samples = jnp.asarray(duration_samples, dtype=xs.dtype)
    duration_samples = jnp.broadcast_to(duration_samples, xs.shape)

    init_state = state

    def body(carry, xs_t):
        st = carry
        x_t, dur_t = xs_t
        y_t, st_next = ramp_smoother_tick(x_t, st, (dur_t,))
        return st_next, y_t

    final_state, ys = lax.scan(body, init_state, (xs, duration_samples))
    return ys, final_state


# =============================================================================
# 3. Allpass smoother
# =============================================================================

def allpass_smoother_init(
    initial_value: float,
    a: float,
    *,
    dtype=jnp.float32,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray]]:
    """
    Initialize allpass smoother.

    Args:
        initial_value : starting output and previous input value
        a             : allpass coefficient (|a| < 1 recommended)
        dtype         : JAX dtype

    Returns:
        state  : (prev_x, prev_y)
        params : (a,)
    """
    v = jnp.asarray(initial_value, dtype=dtype)
    prev_x = v
    prev_y = v
    a_arr = jnp.asarray(a, dtype=dtype)
    state = (prev_x, prev_y)
    params = (a_arr,)
    return state, params


def allpass_smoother_update_state(
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Placeholder update_state for allpass smoother.
    The state is evolved in tick(), so this is a pass-through.
    """
    return state


@jax.jit
def allpass_smoother_tick(
    x: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    First-order allpass smoother single tick.

    y[n] = a * x[n] + prev_x - a * prev_y

    Args:
        x      : input sample
        state  : (prev_x, prev_y)
        params : (a,)

    Returns:
        y        : smoothed output
        new_state: (prev_x_next, prev_y_next)
    """
    prev_x, prev_y = state
    (a,) = params

    x = jnp.asarray(x)
    prev_x = jnp.asarray(prev_x, dtype=x.dtype)
    prev_y = jnp.asarray(prev_y, dtype=x.dtype)
    a = jnp.asarray(a, dtype=x.dtype)

    a = jnp.clip(a, -0.999, 0.999)

    y = a * x + prev_x - a * prev_y

    new_state = (x, y)
    return y, new_state


@jax.jit
def allpass_smoother_process(
    xs: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Process a buffer with the first-order allpass smoother.

    Args:
        xs     : input buffer, shape (T,)
        state  : (prev_x, prev_y)
        params : (a,), scalar or shape (T,)

    Returns:
        ys         : smoothed buffer, shape (T,)
        final_state
    """
    xs = jnp.asarray(xs)
    (a,) = params
    a = jnp.asarray(a, dtype=xs.dtype)
    a = jnp.broadcast_to(a, xs.shape)

    init_state = state

    def body(carry, xs_t):
        st = carry
        x_t, a_t = xs_t
        y_t, st_next = allpass_smoother_tick(x_t, st, (a_t,))
        return st_next, y_t

    final_state, ys = lax.scan(body, init_state, (xs, a))
    return ys, final_state


# =============================================================================
# 4. Smoke tests, plot, listen
# =============================================================================

if __name__ == "__main__":
    import numpy as onp
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

    sr = 48000
    T = 2000

    # ------------------------------
    # One-pole test: smoothing a step
    # ------------------------------
    step = jnp.concatenate([
        jnp.zeros((T // 2,), dtype=jnp.float32),
        jnp.ones((T - T // 2,), dtype=jnp.float32),
    ])

    op_state, op_params = onepole_smoother_init(initial_value=0.0, alpha=0.05)
    step_smoothed, op_state_out = onepole_smoother_process(step, op_state, op_params)

    # ------------------------------
    # Ramp test: ramp from 0 to 1 and back
    # ------------------------------
    ramp_targets = jnp.concatenate([
        jnp.zeros((T // 4,), dtype=jnp.float32),
        jnp.ones((T // 2,), dtype=jnp.float32),
        jnp.zeros((T - 3 * T // 4,), dtype=jnp.float32),
    ])

    rp_state, rp_params = ramp_smoother_init(
        initial_value=0.0,
        initial_target=0.0,
        duration_samples=200.0,
    )
    ramp_out, rp_state_out = ramp_smoother_process(ramp_targets, rp_state, rp_params)

    # ------------------------------
    # Allpass test: smooth white noise
    # ------------------------------
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, (T,), dtype=jnp.float32)

    ap_state, ap_params = allpass_smoother_init(initial_value=0.0, a=0.7)
    noise_smoothed, ap_state_out = allpass_smoother_process(noise, ap_state, ap_params)

    # ------------------------------
    # Plot results
    # ------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(onp.asarray(step), label="step")
    axs[0].plot(onp.asarray(step_smoothed), label="one-pole smoothed")
    axs[0].set_title("One-pole smoother")
    axs[0].legend()

    axs[1].plot(onp.asarray(ramp_targets), label="targets")
    axs[1].plot(onp.asarray(ramp_out), label="ramp output")
    axs[1].set_title("Ramp smoother")
    axs[1].legend()

    axs[2].plot(onp.asarray(noise), alpha=0.3, label="noise")
    axs[2].plot(onp.asarray(noise_smoothed), label="allpass smoothed")
    axs[2].set_title("Allpass smoother")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    # ------------------------------
    # Listen example (optional)
    # ------------------------------
    if HAVE_SD:
        print("Playing noise vs allpass-smoothed noise...")
        # Stack noise and smoothed noise sequentially
        playback = onp.concatenate([
            onp.asarray(noise) * 0.1,
            onp.asarray(noise_smoothed) * 0.1,
        ], axis=0)
        sd.play(playback, samplerate=sr, blocking=True)
        print("Done.")
    else:
        print("sounddevice not available; skipping audio playback.")
```

---

Next natural steps after this module:

* Wire **onepole_smoother** into `phasor_core` for frequency smoothing.
* Use **ramp_smoother** for continuous control changes (cutoff, gain, etc.).
* Use **allpass_smoother** for fractional delay smoothing in delay/reverb cores.
