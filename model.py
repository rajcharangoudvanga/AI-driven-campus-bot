# train_model.py
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

data = json.load(open("intents.json"))
X = []
y = []
for it in data["intents"]:
    for p in it["patterns"]:
        X.append(p)
        y.append(it["tag"])

vec = TfidfVectorizer()
Xv = vec.fit_transform(X)
clf = LogisticRegression(max_iter=500).fit(Xv, y)

joblib.dump(vec, "vec.joblib")
joblib.dump(clf, "clf.joblib")
print("Saved model")
