import re
import nltk
import gradio as gr

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# NLTK SETUP

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))


# FAQ DATASET

faqs = [
    {
        "question": "What is artificial intelligence?",
        "answer": (
            "Artificial Intelligence (AI) is a field of computer science that "
            "enables machines to perform tasks that normally require human "
            "intelligence, such as understanding language, recognizing patterns, "
            "and making decisions."
        )
    },
    {
        "question": "What is AI automation?",
        "answer": (
            "AI automation combines artificial intelligence with automated "
            "workflows to perform tasks with minimal human intervention."
        )
    },
    {
        "question": "What is machine learning?",
        "answer": (
            "Machine learning is a branch of AI where computers learn patterns "
            "from data and use those patterns to make predictions or decisions."
        )
    },
    {
        "question": "What is deep learning?",
        "answer": (
            "Deep learning is a type of machine learning that uses neural networks "
            "with multiple layers to learn complex patterns from large amounts of data."
        )
    },
    {
        "question": "What is NLP?",
        "answer": (
            "Natural Language Processing (NLP) is a field of AI that enables "
            "computers to process, analyze, and understand human language."
        )
    },
    {
        "question": "What is a chatbot?",
        "answer": (
            "A chatbot is a software application that communicates with users "
            "through natural language and can provide information or perform tasks."
        )
    },
    {
        "question": "What is an AI agent?",
        "answer": (
            "An AI agent is a system that can understand information, make "
            "decisions, use tools, and perform actions to accomplish a goal."
        )
    },
    {
        "question": "What is n8n?",
        "answer": (
            "n8n is a workflow automation platform that allows you to connect "
            "applications, APIs, databases, and AI services to automate business processes."
        )
    },
    {
        "question": "What is workflow automation?",
        "answer": (
            "Workflow automation uses software to automatically perform a sequence "
            "of tasks instead of requiring a person to complete every step manually."
        )
    },
    {
        "question": "What is an API?",
        "answer": (
            "An API, or Application Programming Interface, allows different "
            "software applications to communicate and exchange data."
        )
    },
    {
        "question": "What is an AI API?",
        "answer": (
            "An AI API allows a software application to send data to an AI service "
            "and receive results such as text generation, classification, translation, "
            "or analysis."
        )
    },
    {
        "question": "What is TF-IDF?",
        "answer": (
            "TF-IDF stands for Term Frequency-Inverse Document Frequency. "
            "It is a technique that converts text into numerical values based on "
            "how important words are within a document and across a collection of documents."
        )
    },
    {
        "question": "What is cosine similarity?",
        "answer": (
            "Cosine similarity measures how similar two numerical vectors are by "
            "comparing the angle between them. In this chatbot, it is used to compare "
            "the user's question with FAQ questions."
        )
    },
    {
        "question": "How does this FAQ chatbot work?",
        "answer": (
            "The chatbot cleans the user's question, converts it into a TF-IDF vector, "
            "compares it with all FAQ vectors using cosine similarity, and returns the "
            "answer associated with the most similar question."
        )
    },
    {
        "question": "What is text preprocessing?",
        "answer": (
            "Text preprocessing prepares raw text for analysis by performing operations "
            "such as converting text to lowercase, removing punctuation, splitting text "
            "into words, and removing unnecessary stop words."
        )
    },
    {
        "question": "What are stop words?",
        "answer": (
            "Stop words are common words such as 'the', 'is', 'a', and 'and' that often "
            "provide little useful information for text-matching tasks."
        )
    },
    {
        "question": "What is computer vision?",
        "answer": (
            "Computer vision is a field of AI that enables computers to analyze and "
            "understand images and video."
        )
    },
    {
        "question": "What is object detection?",
        "answer": (
            "Object detection is a computer vision technique that identifies objects "
            "in an image or video and determines their locations."
        )
    },
    {
        "question": "What is generative AI?",
        "answer": (
            "Generative AI refers to AI systems that can create new content such as "
            "text, images, audio, video, or code based on learned patterns."
        )
    },
    {
        "question": "What is a neural network?",
        "answer": (
            "A neural network is a machine learning model inspired by the structure "
            "of the brain. It consists of connected nodes organized into layers that process information."
        )
    },
    {
        "question": "What is data preprocessing?",
        "answer": (
            "Data preprocessing is the process of cleaning and transforming raw data "
            "into a format suitable for analysis or machine learning."
        )
    },
    {
        "question": "Why is Python used in AI?",
        "answer": (
            "Python is widely used in AI because it has a simple syntax and a large "
            "ecosystem of libraries for machine learning, data analysis, NLP, computer "
            "vision, and deep learning."
        )
    },
    {
        "question": "What is scikit-learn?",
        "answer": (
            "Scikit-learn is a Python machine learning library that provides tools for "
            "preprocessing, classification, regression, clustering, feature extraction, "
            "and model evaluation."
        )
    },
    {
        "question": "What is NLTK?",
        "answer": (
            "NLTK, or Natural Language Toolkit, is a Python library that provides tools "
            "for working with human language and performing NLP tasks."
        )
    },
    {
        "question": "What is Gradio?",
        "answer": (
            "Gradio is a Python library that makes it easy to build interactive web "
            "interfaces for machine learning and AI applications."
        )
    }
]


# TEXT PREPROCESSING

def preprocess(text):
    """Clean and prepare text for similarity matching."""

    text = text.lower()

    # Remove punctuation, numbers, and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Split into words
    words = text.split()

    # Remove stop words
    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# TF-IDF VECTOR REPRESENTATION

faq_questions = [
    preprocess(faq["question"])
    for faq in faqs
]

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(faq_questions)


# FIND BEST ANSWER

def find_best_answer(user_question):
    """Find the FAQ with the highest cosine similarity."""

    user_question = user_question.strip()

    if not user_question:
        return "Please enter a question.", 0.0

    # Simple greeting handling
    greetings = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    }

    if user_question.lower() in greetings:
        return (
            "Hello! 👋 I'm your AI & Automation FAQ chatbot. "
            "Ask me about AI, machine learning, NLP, n8n, APIs, "
            "automation, or chatbots.",
            1.0
        )

    # Preprocess question
    processed_question = preprocess(user_question)

    # Handle cases where preprocessing removes everything
    if not processed_question:
        return (
            "Please ask a meaningful question related to AI or automation.",
            0.0
        )

    # Convert user question into TF-IDF vector
    user_vector = vectorizer.transform([processed_question])

    # Calculate cosine similarity
    similarities = cosine_similarity(
        user_vector,
        faq_vectors
    )

    # Find best match
    best_index = similarities.argmax()
    best_score = float(similarities[0][best_index])

    # Confidence threshold
    if best_score < 0.20:
        return (
            "I'm sorry, I couldn't find a reliable answer to that question. "
            "Please try asking about AI, machine learning, NLP, n8n, APIs, "
            "automation, or chatbots.",
            best_score
        )

    return faqs[best_index]["answer"], best_score


# CHAT FUNCTION

def chat_with_bot(message, history):
    """Process the user's message and return the chatbot response."""

    if not message or not message.strip():
        return history

    answer, score = find_best_answer(message)

    # Show confidence score in the response
    response = (
        f"{answer}\n\n"
        f"**Similarity score:** {score:.3f}"
    )

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ]

    return history


# ---------------------------------------------------------
# GRADIO INTERFACE
# ---------------------------------------------------------

css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
"""

with gr.Blocks(
    title="AI FAQ Chatbot",
    css=css
) as demo:

    gr.Markdown(
        """
        # 🤖 AI FAQ Chatbot

        ### AI & Automation Knowledge Assistant

        Ask questions about:

        **Artificial Intelligence • Machine Learning • NLP • n8n • APIs • Automation • Chatbots**
        """
    )

    chatbot = gr.Chatbot(
        label="Conversation",
        type="messages",
        height=500
    )

    with gr.Row():

        message = gr.Textbox(
            label="Your Question",
            placeholder="Example: What is AI automation?",
            lines=2,
            scale=4
        )

        ask_button = gr.Button(
            "Ask",
            variant="primary",
            scale=1
        )

    clear_button = gr.Button("Clear Conversation")

    gr.Markdown("### 💡 Try an example")

    gr.Examples(
        examples=[
            "What is artificial intelligence?",
            "What is AI automation?",
            "What is n8n?",
            "What is an AI agent?",
            "What is NLP?",
            "What is machine learning?",
            "What is cosine similarity?",
            "How does this FAQ chatbot work?",
            "What is an API?",
            "What is Gradio?"
        ],
        inputs=message
    )

    gr.Markdown(
        """
        ---
        ### 🧠 How this chatbot works

        **User Question**
        ↓  
        **NLP Preprocessing**
        ↓  
        **TF-IDF Vectorization**
        ↓  
        **Cosine Similarity**
        ↓  
        **Best FAQ Match**
        ↓  
        **Confidence Check**
        ↓  
        **Answer**
        """
    )

    # Send message when Ask is clicked
    ask_button.click(
        chat_with_bot,
        inputs=[message, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[message]
    )

    # Send message when Enter is pressed
    message.submit(
        chat_with_bot,
        inputs=[message, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[message]
    )

    # Clear conversation
    clear_button.click(
        lambda: [],
        outputs=[chatbot]
    )


# ---------------------------------------------------------
# RUN APPLICATION
# ---------------------------------------------------------

if __name__ == "__main__":
    demo.launch()
