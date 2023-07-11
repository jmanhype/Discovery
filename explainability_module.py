class ExplainabilityModule:
    def __init__(self, language_understanding_module, probabilistic_reasoning_module):
        self.language_understanding_module = language_understanding_module
        self.probabilistic_reasoning_module = probabilistic_reasoning_module

    def explain_decision(self, decision, agent, target):
        if target == "human":
            explanation = self.human_friendly_explanation(decision, agent)
        elif target == "machine":
            explanation = self.machine_readable_explanation(decision, agent)
        else:
            raise ValueError("Invalid target for explanation")

        return explanation

    def human_friendly_explanation(self, decision, agent):
        human_friendly_description = self.language_understanding_module.convert_to_human_readable(decision)

        belief_confidence = agent.get_belief_confidence()

        if belief_confidence >= 0.8:
            confidence_message = "I am very confident."
        elif 0.5 <= belief_confidence < 0.8:
            confidence_message = "I am somewhat confident."
        else:
            confidence_message = "I am not very confident."

        explanation = f"I chose to {human_friendly_description} because {confidence_message}"

        return explanation

    def machine_readable_explanation(self, decision, agent):
        plot_expression = self.language_understanding_module.translate_to_plot(decision)
        
        factors = self.probabilistic_reasoning_module.extract_factors(plot_expression)

        confidence = agent.get_belief_confidence()

        explanation = {
            "decision": decision,
            "factors": factors,
            "confidence": confidence
        }
        
        return explanation