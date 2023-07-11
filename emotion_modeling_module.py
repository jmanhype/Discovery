import random
import numpy as np

class EmotionModelingModule:
    def __init__(self):
        self.emotions = ["happy", "sad", "angry", "surprised", "neutral", "anxious", "confused"]
        self.current_state = "neutral"
        self.emotion_transitions = self.initialize_emotion_transitions()

    def initialize_emotion_transitions(self):
        # Initialize empty probability matrix for emotion transitions
        num_emotions = len(self.emotions)
        emotion_transitions = np.zeros((num_emotions, num_emotions))

        # Populate the emotion transitions matrix with random values
        for i in range(num_emotions):
            randomness = [random.uniform(0, 1) for _ in range(num_emotions)]
            total = sum(randomness)
            normalized_randomness = [r / total for r in randomness]
            emotion_transitions[i, :] = normalized_randomness

        return emotion_transitions

    def update_emotion(self, agent_internal_factors, external_stimulus):
        current_emotion_index = self.emotions.index(self.current_state)
        # Determine the next emotion based on internal factors and external stimulus
        next_emotion_probabilities = self.emotion_transitions[current_emotion_index, :] * agent_internal_factors * external_stimulus

        # Normalize probabilities
        total = sum(next_emotion_probabilities)
        normalized_probabilities = [r / total for r in next_emotion_probabilities]

        # Select new emotion based on updated probabilities
        new_emotion = np.random.choice(self.emotions, p=normalized_probabilities)
        self.current_state = new_emotion
        return new_emotion

    def get_current_emotion(self):
        return self.current_state