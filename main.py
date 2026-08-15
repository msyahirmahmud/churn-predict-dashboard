from model import ChurnPredictorModel

def main():
    model = ChurnPredictorModel()
    print("📊 Customer Churn Risk Evaluator")
    print("---------------------------------")
    
    prob = model.predict_churn_probability(tenure_months=3, monthly_charges=89.50, support_tickets=4, paperless_billing=True)
    risk = model.classify_risk(prob)
    
    print(f"Predicted Churn Probability: {prob * 100:.2f}%")
    print(f"Risk Status Classification: {risk}")

if __name__ == "__main__":
    main()
