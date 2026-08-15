"""
Customer Churn Prediction Model & Feature Weight Engine
"""

class ChurnPredictorModel:
    def __init__(self):
        # Logistic Regression feature weights
        self.weights = {
            "tenure_months": -0.05,
            "monthly_charges": 0.03,
            "support_tickets": 0.25,
            "paperless_billing": 0.15
        }
        self.intercept = -1.20

    def predict_churn_probability(self, tenure_months: int, monthly_charges: float, support_tickets: int, paperless_billing: bool) -> float:
        z = (
            self.intercept +
            (tenure_months * self.weights["tenure_months"]) +
            (monthly_charges * self.weights["monthly_charges"]) +
            (support_tickets * self.weights["support_tickets"]) +
            (1.0 if paperless_billing else 0.0) * self.weights["paperless_billing"]
        )
        
        # Sigmoid activation function 1 / (1 + e^-z)
        import math
        probability = 1.0 / (1.0 + math.exp(-z))
        return round(probability, 4)

    def classify_risk(self, probability: float) -> str:
        if probability >= 0.70: return "HIGH RISK"
        elif probability >= 0.40: return "MEDIUM RISK"
        else: return "LOW RISK"
