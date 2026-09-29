# 🤖 AI FAQ Chatbot

An AI-powered Frequently Asked Questions (FAQ) chatbot built using Python and Natural Language Processing (NLP).

This project was developed as part of the **CodeAlpha Artificial Intelligence Internship – Task 2**.

## 📌 Project Overview

The chatbot accepts a user's question, processes the text using NLP techniques, compares it with a predefined FAQ dataset, and returns the most relevant answer.

The project uses **TF-IDF vectorization** and **Cosine Similarity** to identify the FAQ that is most similar to the user's question.

## 🚀 Features

* FAQ-based chatbot
* NLP text preprocessing
* Lowercase conversion and punctuation removal
* Stop-word removal using NLTK
* TF-IDF text vectorization
* Cosine similarity matching
* Confidence threshold for uncertain questions
* Greeting detection
* Interactive Gradio interface
* Example questions for easy testing

## 🧠 How It Works

```text
User Question
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Find Best Matching FAQ
      ↓
Confidence Check
      ↓
Return Answer
```

## 🛠️ Technologies Used

* Python
* NLTK
* Scikit-learn
* Gradio

## 📚 NLP Techniques

### Text Preprocessing

The user's question is cleaned before matching. The chatbot:

1. Converts text to lowercase
2. Removes punctuation and numbers
3. Splits text into words
4. Removes common stop words

### TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) converts FAQ questions into numerical representations based on the importance of their words.

### Cosine Similarity

Cosine similarity compares the numerical representation of the user's question with the FAQ questions and identifies the closest match.

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/CodeAlpha_FAQChatbot.git
```

Move into the project directory:

```bash
cd CodeAlpha_FAQChatbot
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Run:

```bash
python app.py
```

The Gradio interface will launch and provide the chatbot interface.

## 🎯 Example Questions

You can ask:

* What is artificial intelligence?
* What is AI automation?
* What is n8n?
* What is an AI agent?
* What is NLP?
* What is machine learning?
* What is an API?
* What is cosine similarity?
* How does this FAQ chatbot work?

## 🔮 Future Improvements

Possible future improvements include:

* Larger FAQ knowledge base
* Better semantic understanding
* Conversation memory
* Database integration
* Multilingual support
* Voice input and output
* Large Language Model integration

## 👩‍💻 Author

Developed as part of the CodeAlpha AI Internship.

**Project:** FAQ Chatbot
**Task:** CodeAlpha Artificial Intelligence – Task 2

