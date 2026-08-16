import unittest
from predictor import ChurnPredictorModel

class TestChurnPredictorModel(unittest.TestCase):
    def setUp(self):
        self.model = ChurnPredictorModel()

    def test_predict_churn_high_risk_retention_recommendation(self):
        res = self.model.predict_churn(
            tenure_months=2,
            monthly_charges=120.0,
            support_tickets=5,
            is_month_to_month=True
        )
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertTrue("20% annual discount" in res["recommended_action"])

    def test_batch_predict_processes_multiple_customers(self):
        customers = [
            {"id": "cust-1", "tenure_months": 24, "monthly_charges": 40.0, "support_tickets": 0, "is_month_to_month": False},
            {"id": "cust-2", "tenure_months": 1, "monthly_charges": 150.0, "support_tickets": 6, "is_month_to_month": True}
        ]
        results = self.model.batch_predict(customers)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["customer_id"], "cust-1")
        self.assertEqual(results[0]["prediction"]["risk_level"], "LOW")
        self.assertEqual(results[1]["prediction"]["risk_level"], "HIGH")

    def test_predict_churn_low_risk(self):
        res = self.model.predict_churn(
            tenure_months=48,
            monthly_charges=30.0,
            support_tickets=0,
            is_month_to_month=False
        )
        self.assertEqual(res["risk_level"], "LOW")
        self.assertTrue("Healthy retention profile" in res["recommended_action"])

if __name__ == '__main__':
    unittest.main()
