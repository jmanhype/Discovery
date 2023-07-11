import numpy as np
from collections import defaultdict

class PerformanceMetrics:
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'reward': [],
            'belief_confidence': [],
            'goal_completion': []
        })

    def update(self, agents):
        for agent in agents:
            # Update agent's reward metric
            reward = agent.get_reward()
            self.metrics[agent]['reward'].append(reward)

            # Update agent's belief confidence metric
            belief_confidence = agent.get_belief_confidence()
            self.metrics[agent]['belief_confidence'].append(belief_confidence)

            # Update agent's goal completion metric
            goal_completion = agent.get_goal_completion()
            self.metrics[agent]['goal_completion'].append(goal_completion)

    def get_metric(self, agent, metric_name):
        if metric_name in self.metrics[agent]:
            return self.metrics[agent][metric_name]
        else:
            print(f"Invalid metric name: {metric_name}")
            return None

    def calculate_average_metrics(self):
        average_metrics = defaultdict(lambda: {
            'reward': 0,
            'belief_confidence': 0,
            'goal_completion': 0
        })
        
        for agent in self.metrics:
            total_reward = np.sum(self.metrics[agent]['reward'])
            total_belief_confidence = np.sum(self.metrics[agent]['belief_confidence'])
            total_goal_completion = np.sum(self.metrics[agent]['goal_completion'])
            
            num_samples = len(self.metrics[agent]['reward'])
            
            average_metrics[agent]['reward'] = total_reward / num_samples
            average_metrics[agent]['belief_confidence'] = total_belief_confidence / num_samples
            average_metrics[agent]['goal_completion'] = total_goal_completion / num_samples

        return average_metrics

    def generate_report(self):
        average_metrics = self.calculate_average_metrics()
        print("Agent performance report:\n")

        for agent in average_metrics:
            print(f"Agent {agent}:")
            print(f"  - Average reward: {average_metrics[agent]['reward']:.2f}")
            print(f"  - Average belief confidence: {average_metrics[agent]['belief_confidence']:.2f}")
            print(f"  - Average goal completion: {average_metrics[agent]['goal_completion']:.2f}\n")