from flask import Flask, request, jsonify, send_from_directory
import joblib, json
import spacy
nlp = spacy.load("en_core_web_sm")
vec = joblib.load("vec.joblib")
clf = joblib.load("clf.joblib")
kb = json.load(open("kb.json"))

app = Flask(__name__, static_folder="static")

@app.route("/api/message", methods=["POST"])
def message():
    data = request.json
    text = data.get("message","")
    # intent
    x = vec.transform([text])
    tag = clf.predict(x)[0]
    # NER: find possible location tokens
    doc = nlp(text)
    loc = None
    for ent in doc.ents:
        if ent.label_ in ("LOC","ORG","GPE"):
            if ent.text in kb: loc = ent.text; break
    # fallback: check tokens against kb keys
    if not loc:
        for k in kb:
            if k.lower() in text.lower():
                loc = k; break
        if tag == "greeting":
            return jsonify({"reply": "Hello! How can I help you find a place?"})
    if tag=="find_location" and loc:
        info = kb.get(loc)
        return jsonify({"reply":f"{loc} is at {info['building']}, {info['floor']} floor, room {info['room']}.", "loc":info})
    # generic reply
    return jsonify({"reply":"Sorry, I couldn't find that location. Can you rephrase?"})

@app.route("/")
def index():
    return send_from_directory("static","index.html")

if __name__=="__main__":
    app.run(debug=True,port=5000)
