import time
from collections import defaultdict
from probabilistic_reasoning_module import ProbabilisticReasoningModule
from language_understanding_module import LanguageUnderstandingModule
import matplotlib.pyplot as plt

class TheoryOfMind:
    def __init__(self, language_model):
        self.language_understanding_module = LanguageUnderstandingModule(language_model)
        self.probabilistic_reasoning_module = ProbabilisticReasoningModule()
        self.beliefs = defaultdict(list)  # Agent-specific beliefs
        self.belief_confidence = defaultdict(float)  # Confidence for each belief
        self.rewards = defaultdict(int)  # Placeholder for reinforcement learning rewards
        self.agent_profiles = defaultdict(lambda: {'actions': [], 'beliefs': [], 'rewards': []})  # Store historical actions, beliefs and rewards for each agent
        self.belief_uncertainty = defaultdict(lambda: initial_uncertainty)  # Store uncertainty related to each belief

    def interpret_action(self, agent, action):
        # Translate the agent's action to PLoT
        plot_expression = self.language_understanding_module.translate_to_plot(action)

        # Evaluate the PLoT expression to update beliefs
        belief = self.probabilistic_reasoning_module.evaluate_expression(plot_expression)
        self.update_beliefs(agent, belief, confidence=0.5)  # Assuming a default confidence level

        return self.beliefs[agent]

    def update_beliefs(self, agent, new_belief, confidence):
        # Update belief and timestamp it
        self.beliefs[agent].append((time.time(), new_belief))
        self.belief_confidence[agent] = confidence

    def decay_beliefs(self, decay_rate=0.99, threshold=0.01):
        for agent in self.belief_confidence:
            self.belief_confidence[agent] *= decay_rate
            if self.belief_confidence[agent] < threshold:
                self.beliefs[agent].clear()
                self.belief_confidence[agent] = 0

    def update_rewards(self, agent, action, correct_action):
        # Check if the action taken by the agent is the correct action
        if action == correct_action:
            # If the action is correct, increase the agent's reward by 1
            self.rewards[agent] += 1
        else:
            # If the action is incorrect, decrease the agent's reward by 1
            self.rewards[agent] -= 1

    def get_beliefs(self, agent):
        return self.beliefs[agent]

    def get_belief_confidence(self, agent):
        return self.belief_confidence[agent]
    
    def get_rewards(self, agent):
        return self.rewards[agent]

    def update_confidence(self, agent):
        # Check if the agent has any recorded actions and rewards
        if self.agent_profiles[agent]['actions'] and self.agent_profiles[agent]['rewards']:
            # Get the most recent action and reward of the agent
            recent_action = self.agent_profiles[agent]['actions'][-1]
            recent_reward = self.agent_profiles[agent]['rewards'][-1]

            # Update the agent's confidence based on the success of the recent action
            if recent_reward > 0:
                self.belief_confidence[agent] = min(self.belief_confidence[agent] + 0.1, 1.0)
            else:
                self.belief_confidence[agent] = max(self.belief_confidence[agent] - 0.1, 0.0)
        else:
            print(f"No actions or rewards recorded for agent {agent}.")

    def visualize_belief_evolution(self, agent):
        # Check if the agent has any recorded beliefs
        if self.beliefs[agent]:
            # Separate the timestamps and beliefs
            timestamps, beliefs = zip(*self.beliefs[agent])

            # Convert the timestamps to relative time
            relative_timestamps = [t - timestamps[0] for t in timestamps]

            # Plot the beliefs over time
            plt.figure(figsize=(10, 5))
            plt.plot(relative_timestamps, beliefs)
            plt.xlabel('Time')
            plt.ylabel('Belief')
            plt.title(f'Belief evolution for agent {agent}')
            plt.grid(True)
            plt.show()
        else:
            print(f"No beliefs recorded for agent {agent}.")

    def query_other_agents(self, querying_agent, queried_agent):
        # Check if the queried agent has any recorded actions
        if self.agent_profiles[queried_agent]['actions']:
            # Get the most recent action of the queried agent
            recent_action = self.agent_profiles[queried_agent]['actions'][-1]

            # Interpret the action using the querying agent's understanding
            interpreted_action = self.interpret_action(querying_agent, recent_action)

            # Update the querying agent's beliefs based on the interpreted action
            self.update_beliefs(querying_agent, interpreted_action, confidence=0.5)
        else:
            print(f"No actions recorded for agent {queried_agent}.")