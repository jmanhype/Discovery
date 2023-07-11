import random

class SelfImprovementModule:
    def __init__(self):
        self.performance_history = []
        self.learning_rate = 0.1

    def update_performance_history(self, performance):
        self.performance_history.append(performance)

    def self_improve(self, action):
        improved_action = action
        probability_of_improvement = self.calculate_improvement_probability()

        if random.random() < probability_of_improvement:
            # Update the action based on the learning rate
            improved_action = self.apply_learning_rate(action)

        return improved_action

    def calculate_improvement_probability(self):
        if len(self.performance_history) < 2:
            # If there's not enough performance history data, set an arbitrary probability
            return 0.5

        # Calculate the performance improvement rate based on the history data
        last_performance = self.performance_history[-1]
        second_last_performance = self.performance_history[-2]

        if last_performance > second_last_performance:
            performance_improvement_rate = (last_performance - second_last_performance) / second_last_performance
        else:
            performance_improvement_rate = 0.0

        # Calculate the improvement probability using the performance improvement rate
        improvement_probability = 0.5 + performance_improvement_rate
        return improvement_probability

    def apply_learning_rate(self, action):
        # This is a placeholder implementation.
        # You could add your own logic for how the learning rate should affect the action of the agent.
        improved_action = action * self.learning_rate
        return improved_action

    def update_learning_rate(self, new_learning_rate):
        self.learning_rate = new_learning_rate
