"""Reward function for the bin-clearing RL environment.

One RL step = one pick attempt. In ``reward_mode="physics"`` the chosen grasp
pose is executed with real contact physics (see
``control/pick_place_primitive.attempt_grasp_physical``) and the per-step
reward is::

    step_penalty
  + grasp_quality_coef * grasp_quality
  + deliver_reward     if delivered
  - disturb_coef * neighbour_disturbance_m
  - eject_penalty * items_ejected

Special cases (no physics is run):
  * invalid action  -> step_penalty + invalid_penalty
  * empty grab      -> step_penalty + empty_grab_penalty

``reward_mode="geometric"`` (the old fast path, kept for smoke tests) uses the
deterministic geometric predicate. see ``compute_reward_geometric`` below.
"""


def compute_reward(invalid_action: bool,
                   item_present: bool,
                   delivered: bool,
                   outcome: dict = None,
                   step_penalty: float = -0.02,
                   deliver_reward: float = 3.0,
                   grasp_quality_coef: float = 0.30,
                   empty_grab_penalty: float = -0.50,
                   invalid_penalty: float = -1.0,
                   disturb_coef: float = 2.0,
                   eject_penalty: float = 1.0,
                   approach_clear_pred_coef: float = 0.50,
                   approach_clear_pred: float = 0.0,
                   lookahead_bonus_coef: float = 0.20,
                   keystone_penalty_coef: float = 0.30,
                   lookahead_bonus_norm: float = 0.0,
                   cascade_collapse_flag: bool = False) -> float:
    """Physics-grounded shaped reward.

    v5 L4 sequential-reasoning terms:
      + lookahead_bonus_coef * min(1.0, lookahead_bonus_norm)  on delivery
      - keystone_penalty_coef * cascade_collapse_flag          on any valid attempt

    Both terms gate on a real grasp attempt (not invalid). The lookahead bonus
    additionally gates on ``delivered``.

    Args:
        invalid_action: chosen action index had no candidate.
        item_present:   a source-bin item was associated with the chosen grasp.
        delivered:      the grasp held and the item was moved to the dest bin.
        outcome:        dict from ``attempt_grasp_physical`` (``grasp_quality``,
                        ``neighbour_disturbance_m``, ``items_ejected``).
        approach_clear_pred_coef: weight on the predictive approach-clearance
                        shaping term. Policy-invariant under selection-only.
        approach_clear_pred: scalar in [0, 1]. post-refinement cylinder-clearance
                        predicate at the executed grasp pose.
        lookahead_bonus_coef: weight on the sequential lookahead bonus.
        keystone_penalty_coef: weight on the cascade/keystone penalty.
        lookahead_bonus_norm: float in [0, 1]. normalised newly-exposed count.
        cascade_collapse_flag: True iff pile collapsed and items left the bin.
    """
    if invalid_action:
        return float(step_penalty + invalid_penalty)

    o = outcome or {}
    disturb = float(o.get("neighbour_disturbance_m", 0.0))
    ejected = int(o.get("items_ejected", 0))
    approach_bonus = float(approach_clear_pred_coef) * float(approach_clear_pred)

    look_norm = float(lookahead_bonus_norm)
    if look_norm < 0.0:
        look_norm = 0.0
    elif look_norm > 1.0:
        look_norm = 1.0
    lookahead_bonus = (float(lookahead_bonus_coef) * look_norm
                       if delivered else 0.0)
    keystone_penalty = (float(keystone_penalty_coef)
                        if bool(cascade_collapse_flag) else 0.0)

    if not item_present:
        # grabbed empty space, hand may still have swept the pile
        return float(step_penalty + empty_grab_penalty
                     - disturb_coef * disturb - eject_penalty * ejected
                     + approach_bonus
                     - keystone_penalty)

    r = float(step_penalty)
    r += grasp_quality_coef * float(o.get("grasp_quality", 0.0))
    if delivered:
        r += deliver_reward
    r += -disturb_coef * disturb
    r += -eject_penalty * ejected
    r += approach_bonus
    r += lookahead_bonus
    r += -keystone_penalty
    return float(r)


def compute_reward_geometric(success: bool,
                             invalid_action: bool,
                             delivered_one: bool,
                             step_penalty: float = -0.01,
                             success_reward: float = 1.0,
                             fail_penalty: float = -0.2,
                             invalid_penalty: float = -1.0) -> float:
    """Old fast-path reward: deterministic geometric predicate, no physics."""
    r = float(step_penalty)
    if invalid_action:
        return r + float(invalid_penalty)
    if success and delivered_one:
        return r + float(success_reward)
    return r + float(fail_penalty)


def compute_episode_metrics(episode_log: list) -> dict:
    """Aggregate per-step info dicts: ``reward``, ``success``, ``invalid_action``,
    ``n_delivered``, ``terminated``."""
    if not episode_log:
        return {"total_reward": 0.0, "n_delivered": 0, "success_rate": 0.0,
                "invalid_rate": 0.0, "n_steps": 0, "cleared": False}
    n = len(episode_log)
    total = sum(float(s.get("reward", 0.0)) for s in episode_log)
    n_succ = sum(1 for s in episode_log if s.get("success"))
    n_inv = sum(1 for s in episode_log if s.get("invalid_action"))
    n_deliv = episode_log[-1].get("n_delivered", n_succ)
    return {
        "total_reward": total,
        "n_delivered": n_deliv,
        "success_rate": n_succ / n,
        "invalid_rate": n_inv / n,
        "n_steps": n,
        "cleared": bool(episode_log[-1].get("terminated", False)),
    }
