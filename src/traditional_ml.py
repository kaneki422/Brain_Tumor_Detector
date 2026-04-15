from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

MODELS = {
    'SVM':          SVC(kernel='rbf', probability=True),
    'KNN':          KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes':  GaussianNB(),
    'Random Forest':RandomForestClassifier(n_estimators=200),
}

def train_all(X, y, save_dir='models/'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, f'{save_dir}scaler.pkl')

    results = {}
    for name, model in MODELS.items():
        model.fit(X_train_sc, y_train)
        preds = model.predict(X_test_sc)
        acc = accuracy_score(y_test, preds)
        results[name] = {
            'accuracy': acc,
            'report':   classification_report(y_test, preds)
        }
        joblib.dump(model, f'{save_dir}{name.replace(" ","_")}.pkl')
        print(f"{name}: {acc*100:.2f}%")
    return results

def predict_ml(image_features, model_name, model_dir='models/'):
    scaler = joblib.load(f'{model_dir}scaler.pkl')
    model  = joblib.load(f'{model_dir}{model_name.replace(" ","_")}.pkl')
    feat_scaled = scaler.transform([image_features])
    pred  = model.predict(feat_scaled)[0]
    prob  = model.predict_proba(feat_scaled)[0]
    return pred, max(prob)