import random

class OnDevicePersonalization:
    def __init__(self, initial_preferences=None, learning_rate=0.2):
        # Simulate initial model/preferences stored on the device
        self.preferences = initial_preferences if initial_preferences else {
            'action': 0.5,
            'comedy': 0.5,
            'drama': 0.5,
            'sci-fi': 0.5
        }
        self.learning_rate = learning_rate
        print("--- Initial Device Model (User Preferences) ---")
        self._print_preferences()

    def _print_preferences(self):
        for category, score in self.preferences.items():
            print(f"- {category.capitalize()}: {score:.2f}")
        print("-" * 40)

    def simulate_user_interaction(self, item, rating):
        """
        Simulates a user interacting with an item and providing a rating.
        This data is used for on-device training to personalize the model.
        """
        print(f"\nUser rated '{item['name']}' with {rating}/5 stars.")
        print(f"Item features: {item['features']}")

        # --- ON-DEVICE TRAINING STEP ---
        # The core concept: User interaction data stays on the device,
        # and the personalization model updates locally.
        for feature, item_value in item['features'].items():
            if feature in self.preferences:
                # Simple update rule: adjust preference towards/away from item's features
                # based on the rating. (Rating 3 is neutral, 5 is strong like, 1 is strong dislike)
                delta = (rating - 3) / 2.0 # Normalize rating to -1 to 1 range
                self.preferences[feature] += self.learning_rate * delta * item_value
                # Clamp preferences between 0 and 1 to keep them valid scores
                self.preferences[feature] = max(0, min(1, self.preferences[feature]))

        print("--- Device Model Updated ---")
        self._print_preferences()

    def recommend_items(self, available_items, num_recommendations=3):
        """
        Generates recommendations based on the current on-device personalized model.
        """
        print("\n--- Generating Recommendations ---")
        scored_items = []
        for item in available_items:
            score = 0
            for feature, user_pref in self.preferences.items():
                # Score item by multiplying user preference with item's feature value
                score += user_pref * item['features'].get(feature, 0)
            scored_items.append((item, score))

        # Sort by score in descending order to get top recommendations
        scored_items.sort(key=lambda x: x[1], reverse=True)

        print("Top Recommendations:")
        for i, (item, score) in enumerate(scored_items[:num_recommendations]):
            print(f"{i+1}. {item['name']} (Score: {score:.2f})")
        print("-" * 40)


# --- Example Usage ---
if __name__ == "__main__":
    # Define some example items with features (e.g., movies, articles)
    all_items = [
        {'name': 'The Action Hero', 'features': {'action': 1.0, 'comedy': 0.2, 'drama': 0.1, 'sci-fi': 0.0}},
        {'name': 'Laugh Out Loud', 'features': {'action': 0.1, 'comedy': 1.0, 'drama': 0.3, 'sci-fi': 0.0}},
        {'name': 'Deep Thoughts Drama', 'features': {'action': 0.0, 'comedy': 0.1, 'drama': 1.0, 'sci-fi': 0.2}},
        {'name': 'Future Shock', 'features': {'action': 0.8, 'comedy': 0.0, 'drama': 0.2, 'sci-fi': 1.0}},
        {'name': 'Romantic Comedy', 'features': {'action': 0.0, 'comedy': 0.9, 'drama': 0.5, 'sci-fi': 0.0}},
        {'name': 'Epic Sci-Fi Battle', 'features': {'action': 0.9, 'comedy': 0.0, 'drama': 0.1, 'sci-fi': 1.0}},
        {'name': 'Thought-Provoking Indie', 'features': {'action': 0.1, 'comedy': 0.3, 'drama': 0.9, 'sci-fi': 0.4}},
    ]

    # Initialize the on-device personalization system with default preferences
    personalizer = OnDevicePersonalization(learning_rate=0.2)

    # Get initial recommendations based on default preferences
    personalizer.recommend_items(all_items)

    # Simulate user interactions, triggering on-device training
    personalizer.simulate_user_interaction(all_items[0], 5) # User loves 'The Action Hero'
    personalizer.simulate_user_interaction(all_items[2], 2) # User dislikes 'Deep Thoughts Drama'
    personalizer.simulate_user_interaction(all_items[1], 4) # User likes 'Laugh Out Loud'

    # See how recommendations change after the first round of on-device training
    personalizer.recommend_items(all_items)

    # Simulate more interactions to further refine the on-device model
    personalizer.simulate_user_interaction(all_items[3], 5) # User loves 'Future Shock'
    personalizer.simulate_user_interaction(all_items[4], 1) # User hates 'Romantic Comedy'

    # Get final recommendations reflecting the fully personalized on-device model
    personalizer.recommend_items(all_items)
