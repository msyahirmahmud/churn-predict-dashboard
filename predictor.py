"""
Customer Churn Risk Predictor & Analytics Model Engine
"""

class ChurnPredictorModel:
    def __init__(self):
        # Weights for heuristic churn probability estimation
        self.weights = {
            "tenure_months": -0.04,        # Longer tenure = lower churn
            "monthly_charges": 0.015,       # Higher charges = higher churn
            "support_tickets": 0.12,        # More tickets = higher churn
            "contract_is_monthly": 0.25     # Month-to-month = higher churn
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
            "risk_level": risk_level
        }

    def batch_predict(self, customer_records: list) -> list:
        results = []
        for cust in customer_records:
            pred = self.predict_churn(
                cust.get("tenure_months", 12),
                cust.get("monthly_charges", 50.0),
                cust.get("support_tickets", 0),
                cust.get("is_month_to_month", True)
            )
            results.append({
                "customer_id": cust.get("id"),
                "prediction": pred
            })
        return results
