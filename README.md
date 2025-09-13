# 🧭 AI-Driven Smart Assistant for Campus Navigation

## 📌 Project Overview
This project is a **conversational AI assistant** designed to help students and visitors **navigate a campus** by locating classrooms, labs, hostels, canteen, library, and more.  

The assistant can:
- Answer **greetings & casual queries**  
- Provide **room/building details** from a knowledge base  
- Tell the **current time and date**  
- List available **campus facilities**  
- Handle **thanks / goodbye** politely  
- Give **fallback responses** when it doesn’t understand  

It is built with **Flask (backend)**, **scikit-learn (intent classification)**, **spaCy (entity recognition)**, and a simple **JSON knowledge base**. The frontend is a clean **HTML/CSS/JS chat UI**.

---

## ⚙️ Tech Stack
- **Backend**: Flask (Python)  
- **NLP**: scikit-learn (TF-IDF + Logistic Regression), spaCy (NER)  
- **Knowledge Base**: JSON file (`kb.json`)  
- **Frontend**: HTML, CSS, Vanilla JS  
- **Model Persistence**: Joblib (`vec.joblib`, `clf.joblib`)  

---

## 🚀 How to Run Locally

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/campus-navigation-bot.git
   cd "AI-Driven Smart Assistant for Campus Navigation"
