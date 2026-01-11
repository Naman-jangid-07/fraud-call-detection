class FraudScorer:
    def __init__(self):
        self.history = []

    def calculate_rolling_risk(self, current_score):
        self.history.append(current_score)
        if len(self.history) > 5:
            self.history.pop(0)
            
        # Average risk over the last few segments to prevent false alarms
        return sum(self.history) / len(self.history)