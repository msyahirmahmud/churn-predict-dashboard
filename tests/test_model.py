import unittest
from model import ChurnPredictorModel

class TestChurnPredictorModel(unittest.TestCase):
    def setUp(self):
        self.model = ChurnPredictorModel()

    def test_high_risk_customer(self):
        # 1 month tenure, high charges, 5 support tickets -> High risk
        prob = self.model.predict_churn_probability(1, 120.0, 5, True)
        risk = self.model.classify_risk(prob)
        self.assertGreaterEqual(prob, 0.70)
        self.assertEqual(risk, "HIGH RISK")

    def test_low_risk_loyal_customer(self):
        # 48 months tenure, low charges, 0 support tickets -> Low risk
        prob = self.model.predict_churn_probability(48, 35.0, 0, False)
        risk = self.model.classify_risk(prob)
        self.assertLess(prob, 0.40)
        self.assertEqual(risk, "LOW RISK")

if __name__ == '__main__':
    unittest.main()
