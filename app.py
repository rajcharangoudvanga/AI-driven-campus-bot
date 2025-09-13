from flask import Flask, request, jsonify, send_from_directory
import joblib, json
import spacy
import datetime

# Load NLP, models, and knowledge base
nlp = spacy.load("en_core_web_sm")
vec = joblib.load("vec.joblib")
clf = joblib.load("clf.joblib")
kb = json.load(open("kb.json"))

app = Flask(__name__, static_folder="static")

@app.route("/api/message", methods=["POST"])
def message():
    data = request.json
    text = data.get("message", "").strip()

    # Predict intent
    x = vec.transform([text])
    tag = clf.predict(x)[0]

    # ---- Intent Handlers ----
    if tag == "greeting":
        return jsonify({"reply": "Hello! How can I help you find a place?"})

    if tag == "goodbye":
        return jsonify({"reply": "Goodbye! Have a great day!"})

    if tag == "thanks":
        return jsonify({"reply": "You're welcome! Happy to help 😊"})

    if tag == "time":
        now = datetime.datetime.now().strftime("%H:%M:%S")
        return jsonify({"reply": f"The current time is {now}."})

    if tag == "date":
        today = datetime.date.today().strftime("%B %d, %Y")
        return jsonify({"reply": f"Today's date is {today}."})

    if tag == "facilities":
        return jsonify({
            "reply": "We have Library, Auditorium, Labs, Hostel, Canteen, Parking, Sports Complex, Medical Center, and more."
        })

    # ---- Location intent (find_location) ----
    if tag == "find_location":
        loc = None

        # Try NER-based detection
        doc = nlp(text)
        for ent in doc.ents:
            if ent.text in kb:
                loc = ent.text
                break

        # Try keyword-based detection
        if not loc:
            for k in kb:
                if k.lower() in text.lower():
                    loc = k
                    break

        # Respond if location is found
        if loc:
            info = kb.get(loc)
            return jsonify({
                "reply": f"{loc} is at {info['building']}, {info['floor']} floor, room {info['room']}.",
                "loc": info
            })

    # ---- Fallback ----
    return jsonify({"reply": "Sorry, I couldn't find that location. Can you rephrase?"})


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
