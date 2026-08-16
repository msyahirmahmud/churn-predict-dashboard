"""
Customer Churn Risk Predictor & Analytics Model Engine
"""

class ChurnPredictorModel:
    def __init__(self):
        self.weights = {
            "tenure_months": -0.04,
            "monthly_charges": 0.015,
            "support_tickets": 0.12,
            "contract_is_monthly": 0.25
        }

    def predict_churn(self, tenure_months: int, monthly_charges: float, support_tickets: int, is_month_to_month: bool) -> dict:
        base_score = 0.35
        score = base_score + (
            (tenure_months * self.weights["tenure_months"]) +
            (monthly_charges * self.weights["monthly_charges"]) +
            (support_tickets * self.weights["support_tickets"]) +
            (0.25 if is_month_to_month else 0.0)
        )

        probability = max(0.01, min(0.99, round(score, 4)))
        risk_level = "LOW"
        if probability >= 0.70:
            risk_level = "HIGH"
        elif probability >= 0.40:
            risk_level = "MEDIUM"

        return {
            "probability": probability,
            "percentage": f"{round(probability * 100, 2)}%",
            "risk_level": risk_level,
            "recommended_action": self.get_retention_action(risk_level, monthly_charges)
        }

    def get_retention_action(self, risk_level: str, monthly_charges: float) -> str:
        if risk_level == "HIGH":
            if monthly_charges > 100:
                return "Offer 20% annual discount & priority account manager"
            return "Offer 15% discount on 1-year contract extension"
        elif risk_level == "MEDIUM":
            return "Send customer satisfaction survey & feature update guide"
        return "No action needed (Healthy retention profile)"
