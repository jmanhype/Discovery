import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

class AdaptiveLearningModule:
    def __init__(self, initial_strategy='random'):
        self.strategies = ['random', 'greedy', 'epsilon_greedy', 'ucb']
        self.initial_strategy = initial_strategy
        self.strategy_performance = {strategy: [] for strategy in self.strategies}
        self.current_strategy = initial_strategy
        self.clf = None
        self.data = []

    def get_action(self, action_space, q_values, explore_prob=0.1):
        if self.current_strategy == 'random':
            return random.choice(action_space)
        elif self.current_strategy == 'greedy':
            return np.argmax(q_values)
        elif self.current_strategy == 'epsilon_greedy':
            if random.random() < explore_prob:
                return random.choice(action_space)
            else:
                return np.argmax(q_values)
        elif self.current_strategy == 'ucb':
            total_plays = sum(q_values.values())
            exploration_bonus = np.array([np.sqrt(2 * np.log(total_plays) / q_values[a]) for a in action_space])
            q_values_ucb = np.array(q_values) + exploration_bonus
            return np.argmax(q_values_ucb)

    def adapt_strategy(self, strategy, performance, features=None):
        self.strategy_performance[strategy].append(performance)
        if features is not None:
            self.data.append((features, strategy, performance))

        # Check if enough data is available to train
        if len(self.data) > 50:
            self.train_adaptive_model()

    def train_adaptive_model(self):
        x, y_strategies, y_performance = zip(*self.data)

        x_train, x_test, y_train, y_test = train_test_split(x, y_strategies, test_size=0.3, random_state=42)
        self.clf = LogisticRegression(multi_class='multinomial', max_iter=5000)
        self.clf.fit(x_train, y_train)

        y_pred_train = self.clf.predict(x_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_f1_score = f1_score(y_train, y_pred_train, average='macro')

        y_pred_test = self.clf.predict(x_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_f1_score = f1_score(y_test, y_pred_test, average='macro')

        print("Adaptive Learning Model")
        print(f"Train accuracy: {train_accuracy}, Train F1-score: {train_f1_score}")
        print(f"Test accuracy: {test_accuracy}, Test F1-score: {test_f1_score}")

        # Update the current strategy based on the adaptive model
        self.update_current_strategy()

    def update_current_strategy(self):
        if not self.clf:
            self.current_strategy = self.initial_strategy
            return

        current_features = self.get_features_from_world_state()
        strategy_prediction = self.clf.predict(current_features)[0]

        # If prediction confidence is high enough, update the strategy
        if max(self.clf.predict_proba(current_features)[0]) > 0.6:
            self.current_strategy = strategy_prediction
        else:
            self.current_strategy = self.initial_strategy
            print("Fallback to initial strategy due to low prediction confidence.")

    def get_features_from_world_state(self):
        # Extract features from the world state as required for adaptation
        pass