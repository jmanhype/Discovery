from collections import defaultdict

class UncertaintyHandlingModule:
    def __init__(self):
        self.belief_uncertainty = defaultdict(lambda: initial_uncertainty)

    def update_uncertainty(self, agent, action, success):
        """
        agent: the id of the agent taking the action
        action: the action taken by the agent
        reward: a boolean indicating whether the action was successful
        """
        # If the action was successful, then uncertainty decreases and goes towards 0
        if success:
            self.belief_uncertainty[agent][action] = max(self.belief_uncertainty[agent][action] - 0.1, 0.0)
        # If the action was unsuccessful, then uncertainty increases and goes towards 1
        else:
            self.belief_uncertainty[agent][action] = min(self.belief_uncertainty[agent][action] + 0.1, 1.0)

    def get_uncertainty(self, agent, action):
        return self.belief_uncertainty[agent][action]

    def take_action_based_on_uncertainty(self, agent, actions):
        """
        agent: the id of the agent taking the action
        actions: a list of all possible actions the agent can take
        """
        min_uncertainty = float('inf')
        best_action = None

        # Find the action with the minimum uncertainty
        for action in actions:
            uncertainty = self.belief_uncertainty[agent][action]
            if uncertainty < min_uncertainty:
                min_uncertainty = uncertainty
                best_action = action

        return best_action

    def choose_least_uncertain_agent(self, agents):
        """
        agents: a list of agent ids
        """
        least_uncertain_agent = None
        min_avg_uncertainty = float('inf')

        # Find the agent with the minimum average uncertainty
        for agent in agents:
            avg_uncertainty = sum(self.belief_uncertainty[agent].values()) / len(self.belief_uncertainty[agent])
            if avg_uncertainty < min_avg_uncertainty:
                min_avg_uncertainty = avg_uncertainty
                least_uncertain_agent = agent

        return least_uncertain_agent