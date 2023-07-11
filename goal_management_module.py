from collections import defaultdict

class GoalManagementModule:

    def __init__(self):
        self.goals = defaultdict(list)
        self.active_goals = defaultdict(list)
        self.current_goals = ["goal1", "goal2", "goal3"]  # Add default goals here

    def add_goal(self, agent, goal):
        self.goals[agent].append(goal)

    def remove_goal(self, agent, goal):
        if goal in self.goals[agent]:
            self.goals[agent].remove(goal)
        else:
            print(f"Goal '{goal}' not found for agent {agent}.")

    def activate_goal(self, agent, goal):
        if goal in self.goals[agent] and goal not in self.active_goals[agent]:
            self.active_goals[agent].append(goal)
        else:
            print(f"Goal '{goal}' not found or already active for agent {agent}.")

    def deactivate_goal(self, agent, goal):
        if goal in self.active_goals[agent]:
            self.active_goals[agent].remove(goal)
        else:
            print(f"Goal '{goal}' not found or not active for agent {agent}.")

    def get_goals(self, agent):
        return self.goals[agent]

    def get_active_goals(self, agent):
        return self.active_goals[agent]