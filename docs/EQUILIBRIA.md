# Equilibria: what is measured, and why

Contribution 3 asks for equilibria alongside the sunbursts. This note fixes which notion the code measures (PLAN.md W8).

## Which notion

A finitely nested I-POMDP agent best-responds to its *beliefs* about the opponent. It is not solving for a Nash equilibrium, and in general it cannot reach one:

- An agent of level l puts zero prior mass on models of level ≥ l. An opponent of equal or higher level therefore lies outside the agent's support. This is the absence of a "grain of truth" (Kalai & Lehrer 1993).
- Doshi & Gmytrasiewicz (2006), *On the Difficulty of Achieving Equilibrium in Interactive POMDPs* (AAAI), show what learning in I-POMDPs can reach when the prior is compatible with the truth: **subjective equilibrium**. Each agent's predictions of the opponent's future play merge with the play actually realized. They also show that this compatibility condition is hard to meet with finite nesting, so even subjective equilibrium need not be reached.

So the measurable question is not "is play a Nash equilibrium?" but **"do each agent's predictions of its opponent converge to what the opponent actually does along the realized path?"** That is the criterion of Kalai & Lehrer and of Doshi & Gmytrasiewicz. In our studies the real opponent is often outside the agent's support: the real i (L3) lies outside j-L2's support. So non-convergence is an expected and reportable outcome, not a bug.

## What the code records

At every step of an episode (`experiments.episodes`), each intentional agent's predicted distribution of the opponent's next action is recorded as `prediction_<agent>`:

    p_i(a_j) = sum over (s, m_j) of b_i(s, m_j) pi_j(a_j | m_j)

It uses the same modeled policies the agent plans against. The opponent's actual action is `action_<other>` in the same row.

## Metrics (computed in analysis, per step, over trials)

- **Prediction loss:** −log p_i(a_j actual), with a bootstrap interval over trials. It falls to 0 under merging with deterministic play.
- **Hit rate:** the share of steps where the actual action is the predicted mode.
- **Level posterior** (`prob_l<k>_<other>`, already recorded): whether belief concentrates on the opponent's true level when it lies inside the support (j-L2 modeled by i-L3), or on the level that best predicts its play when it does not.

**Reference point:** a uniform predictor has loss log 3 (Tiger has three actions). Merging shows as loss falling well below that and staying there.

## Not claimed

- No Nash or Bayes-Nash equilibrium is computed. Computing one for the two-agent Tiger at these horizons would need a separate equilibrium solver, and the literature above says I-POMDP agents need not play one.
- If a reviewer asks for it, the cheapest addition is a check on realized joint policies: is either agent's realized action a best response to the other's realized policy? That comparison needs an exact model of the opponent's actual policy, which exists only for the L0, L1 and L2 agents at small horizons.
