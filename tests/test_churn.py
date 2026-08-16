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
