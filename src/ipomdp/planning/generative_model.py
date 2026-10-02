"""One simulated joint step: the opponent acts, the world moves, minds advance.

Given a particle (s, m_j) and the planning agent's action a_i:

1. a_j is drawn from the opponent's modeled policy at m_j (uniform for L0;
   for intentional models, the bank's cached policy for m_j's belief);
2. s' is drawn from the transition kernel;
3. if the opponent is intentional and the episode continues, its private
   observation o_j is drawn and m_j advances by its own subjective update
   U_j(b_j, a_j, o_j).

The opponent's update receives only its own action and observation, never the
outer state or the planning agent's action. o_j is sampled once here; its
likelihood is not multiplied in again, because the subjective update already
marginalises everything the opponent cannot see.
"""

from ipomdp.belief.finite_belief import InteractiveState, MentalModel


class InteractiveGenerativeModel:
    def __init__(self, solver_bank):
        self.solver_bank = solver_bank

    def tree_step(self, particle, action, agent_id, physics, rng):
        """Return (next particle, joint action, agent's reward, terminal)."""
        opponent = particle.opponent
        other = opponent.frame.agent_id
        actions, weights = zip(*self.solver_bank.filter.action_distribution(opponent))
        joint = {agent_id: action, other: rng.choices(actions, weights=weights, k=1)[0]}
        following = physics.sample_transition(particle.state, joint, rng)
        reward = physics.get_reward(particle.state, joint, following, agent_id)
        terminal = physics.is_terminal(following)
        # No future decision follows termination, so no private model needs advancing.
        if opponent.frame.level > 0 and not terminal:
            private = physics.sample_observation(following, joint, other, rng)
            posterior = self.solver_bank.filter.update(
                opponent, joint[other], private, terminal=False
            )
            opponent = MentalModel(opponent.frame, posterior.belief)
        return InteractiveState(following, opponent), joint, reward, terminal
