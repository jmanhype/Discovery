import time

class ContextAwarenessModule:
    def __init__(self):
        self.contexts = {}
        self.timestamps = {}
        self.default_context = 'global'

    def update_context(self, agent, context):
        # Update the agent's current context
        self.contexts[agent] = context

        # Update the agent's context timestamp
        self.timestamps[agent] = time.time()

    def get_context(self, agent):
        # Return the agent's current context
        return self.contexts.get(agent, self.default_context)

    def elapsed_time_in_context(self, agent, current_time=None):
        # Check if the agent has a recorded context timestamp
        if agent in self.timestamps:
            # Calculate and return the elapsed time in the current context
            if current_time is None:
                current_time = time.time()
            return current_time - self.timestamps[agent]
        else:
            # If the agent has no recorded context timestamp, return 0
            return 0

    def decay_context(self, agent, decay_rate=0.99, threshold=0.01):
        # Decay the agent's time-stamped context
        elapsed_time = self.elapsed_time_in_context(agent)
        decayed_context_value = elapsed_time * decay_rate

        # If the context value falls below the threshold, reset the agent's context to the default context
        if decayed_context_value < threshold:
            self.update_context(agent, self.default_context)