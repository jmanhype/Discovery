class WorldState:
    def __init__(self):
        self.state_history = []
        self.current_state = {}
        
    def update(self, agents):
        # Update the world state based on the actions of the agents
        agent_positions = {}
        agent_actions = {}

        for agent in agents:
            agent_positions[agent.id] = agent.get_position()
            agent_actions[agent.id] = agent.get_action()

        self.current_state["positions"] = agent_positions
        self.current_state["actions"] = agent_actions
        
        # Add updated current state to state history
        self.state_history.append(self.current_state.copy())

    def get_agent_positions(self):
        return self.current_state["positions"]

    def get_agent_actions(self):
        return self.current_state["actions"]