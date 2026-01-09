import joblib
feature_names = joblib.load('feature_names.pkl')
for i, feat in enumerate(feature_names):
    print(f"{i+1}. {feat}")