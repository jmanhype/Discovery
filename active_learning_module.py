import random

class ActiveLearningModule:
    def __init__(self):
        self.relevant_information = []  # Placeholder for relevant information sources
        self.weights = {}
        self.learning_rate = 0.1
        self.sample_count = {}

    def update_weights(self, information_source, feedback):
        # Given a positive or negative feedback, adjust the weight of the information source
        if feedback == "positive":
            self.weights[information_source] += self.learning_rate
        elif feedback == "negative":
            self.weights[information_source] -= self.learning_rate
        else:
            print(f"Unknown feedback: '{feedback}', expected 'positive' or 'negative'.")

        # Normalize the weights such that they sum to 1
        weight_sum = sum(self.weights.values())
        for source in self.weights:
            self.weights[source] /= weight_sum

    def acquire_information(self):
        # Choose an information source based on the current weights
        if not self.weights:
            info_source = random.choice(self.relevant_information)
            self.sample_count[info_source] = 0
        else:
            item_weights = [self.weights[item] for item in self.relevant_information]
            info_source = random.choices(self.relevant_information, weights=item_weights, k=1)[0]

        # Access the chosen information source and update its sample count
        info = self.access_information_source(info_source)
        self.sample_count[info_source] += 1

        return info

    def access_information_source(self, source):
        # Access the chosen information source, this function should include actual handling with the source
        # (e.g., querying API or fetching data from the source)

        # For demonstration purposes, we'll just return the source name along with a random integer
        return f"{source}_{random.randint(1, 100)}"

    def evaluate_information(self, info):
        # Evaluate the utility of the acquired information using the agent's specific evaluation function
        utility = random.uniform(0, 1)  # For demo purposes, return a random utility value between 0 and 1

        return utility

    def balance_exploration_and_exploitation(self, exploration_constant=1):
        for source in self.sample_count:
            # Calculate the exploration factor
            exploration_factor = exploration_constant * ((self.sample_count[source] + 1) ** 0.5)

            # Update the weight of the information source
            self.weights[source] *= ((1 - exploration_factor) + exploration_factor * random.uniform(0, 1))

        # Normalize weights
        weight_sum = sum(self.weights.values())
        for source in self.weights:
            self.weights[source] /= weight_sum

    def learn(self, feedback=None):
        # Acquire information from a weighted choice of information sources
        info = self.acquire_information()

        # Evaluate the utility of the acquired information
        utility = self.evaluate_information(info)

        # Update the weights of information sources
        self.update_weights(info, feedback)

        # Balance exploration and exploitation of information sources
        self.balance_exploration_and_exploitation()

        return info, utility