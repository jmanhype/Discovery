import numpy as np

class LongTermPlanningModule:
    def __init__(self):
        self.plan_length = 10  # The number of steps in the long-term plan
        self.action_space = [] # List of possible actions
        self.state_space = []  # List of possible states
        self.transition_probabilities = None  # Transition probabilities between states given actions

    def build_transition_probabilities(self, experiences):
        """
        This method constructs the transition probabilities matrix based on the agent's experiences.
        :param experiences: A list of tuples representing the agent's experiences.
                            Each tuple contains (current_state, action, next_state, reward).
        """
        num_states = len(self.state_space)
        num_actions = len(self.action_space)

        state_action_count = np.zeros((num_states, num_actions))
        state_action_state_count = np.zeros((num_states, num_actions, num_states))

        for experience in experiences:
            current_state, action, next_state, _ = experience
            current_state_index = self.state_space.index(current_state)
            action_index = self.action_space.index(action)
            next_state_index = self.state_space.index(next_state)

            state_action_count[current_state_index, action_index] += 1
            state_action_state_count[current_state_index, action_index, next_state_index] += 1

        self.transition_probabilities = state_action_state_count / np.maximum(state_action_count[:, :, np.newaxis], 1)

    def long_term_planning(self, current_state, goal_state, horizon=None):
        if horizon is None:
            horizon = self.plan_length

        state_index = self.state_space.index(current_state)
        goal_index = self.state_space.index(goal_state)

        # Initialize the value function
        value_function = np.zeros((len(self.state_space), horizon + 1))

        # Initialize the policy matrix
        policy = np.zeros((len(self.state_space), horizon), dtype=int)

        for h in range(horizon, 0, -1):
            for s in range(len(self.state_space)):
                max_value = float('-inf')
                max_action = 0
                for a in range(len(self.action_space)):
                    expected_value = np.sum(self.transition_probabilities[s, a, :] * value_function[:, h - 1])
                    if expected_value > max_value:
                        max_value = expected_value
                        max_action = a
                value_function[s, h - 1] = max_value
                policy[s, h - 1] = max_action

        # Get the first action suggested by the policy
        action_index = policy[state_index, 0]
        suggested_action = self.action_space[action_index]

        return suggested_action

    def add_experience(self, current_state, action, next_state, reward):
        """
        This method adds a new experience to the agent's memory.
        The experience is a tuple containing (current_state, action, next_state, reward).
        """
        experience = (current_state, action, next_state, reward)
        self.experiences.append(experience)