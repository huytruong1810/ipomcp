"""Sample physical events, then recursively update private subjective beliefs.

Private observations are sampled once. Their likelihood is NOT multiplied again;
the separate conditioning kernel enumerates hidden events inside the opponent's
own subjective model. No hidden outer state/action is supplied to that update.
"""

import random

from ipomdp.finite_belief import InteractiveState, MentalModel


class InteractiveGenerativeModel:
    def __init__(self, solver_bank):
        self.solver_bank = solver_bank

    def sample_event(self, particle, action_i, agent_id_i, physics_model):
        opponent = particle.opponent
        other = opponent.frame.agent_id
        distribution = self.solver_bank.filter.action_distribution(opponent, particle.state)
        actions, weights = zip(*distribution)
        action_j = random.choices(actions, weights=weights, k=1)[0]
        joint = {agent_id_i: action_i, other: action_j}
        following = physics_model.sample_transition(particle.state, joint)
        reward = physics_model.get_reward(particle.state, joint, following, agent_id_i)
        terminal = physics_model.is_terminal(following)
        return following, joint, reward, terminal

    def tree_step(self, particle, action_i, agent_id_i, physics_model):
        following, joint, reward, terminal = self.sample_event(
            particle, action_i, agent_id_i, physics_model
        )
        opponent = particle.opponent
        other = opponent.frame.agent_id
        action_j = joint[other]
        # No future decision follows termination, so no private policy needs solving.
        if opponent.frame.level > 0 and not terminal:
            private = physics_model.sample_observation(following, joint, other)
            posterior = self.solver_bank.filter.update(opponent, action_j, private, terminal=False)
            opponent = MentalModel(opponent.frame, posterior.belief)
        return InteractiveState(following, opponent), joint, reward, terminal
