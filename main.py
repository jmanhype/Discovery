from agent import Agent
from simulation import Simulation
from theory_of_mind import TheoryOfMind
from language_understanding_module import LanguageUnderstandingModule
from probabilistic_reasoning_module import ProbabilisticReasoningModule
import openai
import random

def main():
    # Initialize OpenAI's API
    openai.api_key = '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'
        
    # Initialize the language understanding module with the api_key
    language_module = LanguageUnderstandingModule(openai.api_key)
    probabilistic_module = ProbabilisticReasoningModule()

    # Initialize the simulation with the specified number of agents
    agent_count = 10
    simulation = Simulation(openai.api_key, agent_count)
    simulation.execute()

    # Begin the simulation
    while not simulation.termination_condition():
        # Update the state of the simulation
        simulation.update_simulation_state()

        # Get the list of all agents
        agents = simulation.agents

        # Iterate through agents and observe others
        for agent in agents:
            # Find other agents that are not the current agent
            other_agents = [a for a in agents if a != agent]

            # If there are other agents, let the current agent observe them
            if other_agents:
                # Randomly select an agent for the current agent to observe
                observed_agent = random.choice(other_agents)

                # Update the agent's beliefs and Theory of Mind
                agent.theory_of_mind.observe(observed_agent)

        # Update the visualization of the simulation
        simulation.gui.update(simulation.world_state, agents)

    # Print the final state of the simulation
    print(simulation.world_state)

if __name__ == "__main__":
    main()