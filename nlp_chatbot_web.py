"""
CSE3246 - NLP Assignment
Chatbot using NLP — Web Interface (Flask)

Tasks covered:
  1. Intent-based chatbot using Python and NLP preprocessing
  2. Answers queries about NLP concepts and AI applications
  3. Modern web UI with Flask backend

Approach:
  - Intent classification via TF-IDF vectorization + cosine similarity
  - NLP preprocessing: tokenization, lowercasing, stopword removal, lemmatization
  - Flask web server with REST API for chat interaction
"""

import re
import string
import random

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, render_template, request, jsonify

# ── NLTK downloads ──────────────────────────────────────────────────────────
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(resource, quiet=True)

# ── Preprocessing ───────────────────────────────────────────────────────────

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

KEEP_WORDS = {"what", "how", "why", "which", "who", "when", "where", "can", "do", "does", "is", "are"}


def preprocess(text):
    """Lowercase, remove punctuation, tokenize, remove stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words or tok in KEEP_WORDS
    ]
    return " ".join(tokens)


# ── Knowledge base ──────────────────────────────────────────────────────────

INTENTS = {
    "greeting": {
        "patterns": [
            "hello", "hi", "hey", "good morning", "good evening",
            "what's up", "howdy", "greetings", "hi there",
        ],
        "responses": [
            "Hello! I'm your NLP chatbot. Ask me anything about NLP or AI!",
            "Hey there! Ready to talk about Natural Language Processing and AI.",
            "Hi! How can I help you with NLP or AI today?",
        ],
    },
    "goodbye": {
        "patterns": [
            "bye", "goodbye", "see you", "exit", "quit", "take care",
            "see you later", "thanks bye",
        ],
        "responses": [
            "Goodbye! Happy learning!",
            "See you later! Keep exploring NLP and AI.",
            "Bye! Feel free to come back anytime.",
        ],
    },
    "thanks": {
        "patterns": [
            "thanks", "thank you", "appreciate it", "thanks a lot",
            "that was helpful", "thanks for the help",
        ],
        "responses": [
            "You're welcome! Ask me anything else about NLP or AI.",
            "Happy to help! Got more questions?",
            "Anytime! That's what I'm here for.",
        ],
    },
    "what_is_nlp": {
        "patterns": [
            "what is nlp", "what is natural language processing",
            "define nlp", "explain nlp", "tell me about nlp",
            "what does nlp mean", "nlp meaning",
        ],
        "responses": [
            "Natural Language Processing (NLP) is a branch of AI that deals with "
            "the interaction between computers and human language. It enables machines "
            "to read, understand, and generate text or speech.",
        ],
    },
    "tokenization": {
        "patterns": [
            "what is tokenization", "explain tokenization",
            "how does tokenization work", "what does tokenize mean",
            "tokenization in nlp", "what are tokens in nlp",
        ],
        "responses": [
            "Tokenization is the process of splitting text into smaller units called "
            "tokens \u2014 typically words or subwords. For example, 'I love NLP' becomes "
            "['I', 'love', 'NLP']. It is usually the first step in any NLP pipeline.",
        ],
    },
    "stemming": {
        "patterns": [
            "what is stemming", "explain stemming", "how does stemming work",
            "stemming in nlp", "difference between stemming and lemmatization",
            "stemming vs lemmatization",
        ],
        "responses": [
            "Stemming reduces words to their root form by chopping off suffixes "
            "(e.g., 'running' \u2192 'run', 'studies' \u2192 'studi'). It is fast but can "
            "produce non-dictionary words. Lemmatization, by contrast, uses vocabulary "
            "and morphological analysis to return proper base forms (e.g., 'better' \u2192 'good').",
        ],
    },
    "lemmatization": {
        "patterns": [
            "what is lemmatization", "explain lemmatization",
            "how does lemmatization work", "lemmatization in nlp",
        ],
        "responses": [
            "Lemmatization maps each word to its dictionary base form (lemma). "
            "Unlike stemming, it considers the context and part of speech, so "
            "'better' becomes 'good' and 'ran' becomes 'run'. NLTK's WordNetLemmatizer "
            "and spaCy both provide lemmatization.",
        ],
    },
    "stopwords": {
        "patterns": [
            "what are stopwords", "explain stopwords", "stopword removal",
            "why remove stopwords", "what is stopword removal",
            "examples of stopwords",
        ],
        "responses": [
            "Stopwords are common words (like 'the', 'is', 'and') that carry little "
            "meaning on their own. Removing them reduces noise and speeds up processing. "
            "NLTK provides a built-in list of English stopwords via nltk.corpus.stopwords.",
        ],
    },
    "pos_tagging": {
        "patterns": [
            "what is pos tagging", "explain part of speech tagging",
            "pos tagging in nlp", "what is part of speech tagging",
            "how does pos tagging work",
        ],
        "responses": [
            "Part-of-Speech (POS) tagging assigns grammatical labels (noun, verb, "
            "adjective, etc.) to each word in a sentence. For example, in 'The cat sat', "
            "'The' is a determiner, 'cat' is a noun, and 'sat' is a verb. Libraries "
            "like NLTK and spaCy provide POS taggers.",
        ],
    },
    "ner": {
        "patterns": [
            "what is ner", "what is named entity recognition",
            "explain named entity recognition", "ner in nlp",
            "how does ner work", "what are named entities",
        ],
        "responses": [
            "Named Entity Recognition (NER) identifies and classifies named entities "
            "in text into categories like Person, Organization, Location, Date, etc. "
            "For example, in 'Apple was founded by Steve Jobs in Cupertino', NER would "
            "tag Apple as ORG, Steve Jobs as PERSON, and Cupertino as LOC.",
        ],
    },
    "sentiment_analysis": {
        "patterns": [
            "what is sentiment analysis", "explain sentiment analysis",
            "how does sentiment analysis work", "sentiment analysis in nlp",
            "what is opinion mining", "how to detect sentiment in text",
        ],
        "responses": [
            "Sentiment analysis (opinion mining) determines the emotional tone of text \u2014 "
            "positive, negative, or neutral. It is widely used for product reviews, social "
            "media monitoring, and customer feedback. Common approaches include lexicon-based "
            "methods and machine-learning classifiers like Naive Bayes or deep learning models.",
        ],
    },
    "tfidf": {
        "patterns": [
            "what is tfidf", "what is tf-idf", "explain tfidf",
            "how does tfidf work", "tf idf in nlp",
            "term frequency inverse document frequency",
        ],
        "responses": [
            "TF-IDF (Term Frequency\u2013Inverse Document Frequency) is a numerical statistic "
            "that reflects how important a word is to a document in a collection. TF measures "
            "how often a term appears in a document, while IDF down-weights terms that appear "
            "in many documents. It is commonly used for text feature extraction.",
        ],
    },
    "word_embeddings": {
        "patterns": [
            "what are word embeddings", "explain word embeddings",
            "word2vec", "what is word2vec", "explain word2vec",
            "glove embeddings", "what is glove",
            "how do word embeddings work",
        ],
        "responses": [
            "Word embeddings represent words as dense vectors in a continuous space, "
            "where semantically similar words are closer together. Word2Vec (by Google) "
            "and GloVe (by Stanford) are popular methods. For example, the vector for "
            "'king' minus 'man' plus 'woman' is close to 'queen'.",
        ],
    },
    "bag_of_words": {
        "patterns": [
            "what is bag of words", "explain bag of words",
            "bow model", "how does bag of words work",
            "bag of words in nlp",
        ],
        "responses": [
            "Bag of Words (BoW) is a text representation that counts the frequency of "
            "each word in a document, ignoring grammar and word order. Each document "
            "becomes a fixed-length vector of word counts. It is simple and effective "
            "for many classification tasks, though it loses word-order information.",
        ],
    },
    "text_classification": {
        "patterns": [
            "what is text classification", "explain text classification",
            "text categorization", "how does text classification work",
            "document classification",
        ],
        "responses": [
            "Text classification assigns predefined labels to text documents. Examples "
            "include spam detection (spam/ham), sentiment analysis (positive/negative), "
            "and topic categorization. Common algorithms include Naive Bayes, SVM, "
            "logistic regression, and deep learning models like BERT.",
        ],
    },
    "language_model": {
        "patterns": [
            "what is a language model", "explain language models",
            "how do language models work", "what are language models used for",
            "n-gram language model",
        ],
        "responses": [
            "A language model estimates the probability of a sequence of words. "
            "Traditional models use n-grams (bigrams, trigrams), while modern ones "
            "like GPT and BERT use deep neural networks (Transformers). They power "
            "text generation, autocomplete, machine translation, and chatbots.",
        ],
    },
    "transformers": {
        "patterns": [
            "what are transformers", "explain transformer model",
            "how do transformers work", "transformer architecture",
            "what is the transformer in nlp", "self attention mechanism",
        ],
        "responses": [
            "Transformers are a deep learning architecture introduced in the paper "
            "'Attention Is All You Need' (2017). They use self-attention to process "
            "all words in a sentence simultaneously (not sequentially like RNNs). "
            "BERT, GPT, and T5 are all based on the Transformer architecture.",
        ],
    },
    "bert": {
        "patterns": [
            "what is bert", "explain bert", "how does bert work",
            "bert in nlp", "what does bert stand for",
        ],
        "responses": [
            "BERT (Bidirectional Encoder Representations from Transformers) is a "
            "pre-trained language model by Google (2018). It reads text in both "
            "directions (left-to-right and right-to-left) to capture context. "
            "BERT is fine-tuned for tasks like question answering, NER, and "
            "sentiment analysis, and set new benchmarks across many NLP tasks.",
        ],
    },
    "machine_translation": {
        "patterns": [
            "what is machine translation", "explain machine translation",
            "how does machine translation work", "neural machine translation",
            "how does google translate work",
        ],
        "responses": [
            "Machine Translation (MT) automatically translates text from one language "
            "to another. Modern Neural MT (NMT) uses encoder-decoder Transformer models. "
            "Google Translate, for example, uses a large NMT system. Challenges include "
            "handling idioms, low-resource languages, and maintaining context.",
        ],
    },
    "text_summarization": {
        "patterns": [
            "what is text summarization", "explain text summarization",
            "how does text summarization work", "extractive vs abstractive summarization",
            "automatic summarization",
        ],
        "responses": [
            "Text summarization condenses a long document into a shorter version. "
            "Extractive summarization selects key sentences from the original text, "
            "while abstractive summarization generates new sentences that capture the "
            "main ideas. Models like BART and T5 are used for abstractive summarization.",
        ],
    },
    "chatbot_types": {
        "patterns": [
            "what are types of chatbots", "rule based vs ai chatbot",
            "types of chatbots", "how do chatbots work",
            "explain chatbot types", "intent based chatbot",
        ],
        "responses": [
            "Chatbots are broadly classified into: (1) Rule-based / Intent-based \u2014 "
            "they match user input to predefined intents using pattern matching or "
            "ML classifiers; (2) Retrieval-based \u2014 they pick the best response from "
            "a knowledge base; (3) Generative \u2014 they use language models (like GPT) "
            "to produce novel responses. This chatbot is an intent-based one using "
            "TF-IDF and cosine similarity.",
        ],
    },
    "ai_in_healthcare": {
        "patterns": [
            "ai in healthcare", "how is ai used in healthcare",
            "artificial intelligence in medicine", "ai medical applications",
            "ai applications in healthcare",
        ],
        "responses": [
            "AI in healthcare includes medical image analysis (X-rays, MRIs), "
            "drug discovery, clinical NLP (extracting info from medical records), "
            "virtual health assistants, and predictive analytics for patient outcomes. "
            "NLP specifically helps extract information from unstructured clinical notes.",
        ],
    },
    "ai_in_education": {
        "patterns": [
            "ai in education", "how is ai used in education",
            "ai applications in education", "ai tutoring",
        ],
        "responses": [
            "AI in education powers intelligent tutoring systems, automated grading, "
            "personalized learning paths, plagiarism detection, and chatbots for "
            "student support. NLP is key for essay scoring, question generation, "
            "and language learning apps.",
        ],
    },
    "ai_in_finance": {
        "patterns": [
            "ai in finance", "how is ai used in finance",
            "ai applications in finance", "ai in banking",
        ],
        "responses": [
            "AI in finance is used for fraud detection, algorithmic trading, "
            "credit scoring, customer service chatbots, risk assessment, and "
            "sentiment analysis of financial news. NLP helps analyze earnings "
            "calls, regulatory filings, and market sentiment.",
        ],
    },
    "ai_in_robotics": {
        "patterns": [
            "ai in robotics", "how is ai used in robotics",
            "ai applications in robotics", "intelligent robots",
        ],
        "responses": [
            "AI in robotics enables autonomous navigation, object recognition, "
            "natural language interaction, and decision-making. NLP allows robots "
            "to understand voice commands and communicate with humans. Examples "
            "include warehouse robots, surgical robots, and home assistants.",
        ],
    },
    "ai_applications_general": {
        "patterns": [
            "what are applications of ai", "ai applications",
            "uses of artificial intelligence", "real world ai applications",
            "how is ai used in real life", "examples of ai",
        ],
        "responses": [
            "AI has wide-ranging applications: virtual assistants (Siri, Alexa), "
            "self-driving cars, recommendation systems (Netflix, YouTube), fraud "
            "detection, medical diagnosis, language translation, image recognition, "
            "and game playing (AlphaGo). NLP specifically powers search engines, "
            "chatbots, text analytics, and machine translation.",
        ],
    },
    "speech_recognition": {
        "patterns": [
            "what is speech recognition", "explain speech recognition",
            "how does speech recognition work", "voice recognition ai",
            "speech to text", "how does siri work", "how does alexa work",
        ],
        "responses": [
            "Speech recognition converts spoken language into text. It uses "
            "acoustic models and language models, often based on deep learning. "
            "Modern systems like Whisper (OpenAI) use Transformer architectures. "
            "Applications include virtual assistants (Siri, Alexa), transcription "
            "services, and voice-controlled devices.",
        ],
    },
    "computer_vision": {
        "patterns": [
            "what is computer vision", "explain computer vision",
            "ai image recognition", "how does computer vision work",
            "object detection ai",
        ],
        "responses": [
            "Computer Vision enables machines to interpret visual information from "
            "images and videos. Key tasks include image classification, object "
            "detection, facial recognition, and image segmentation. CNNs and "
            "Vision Transformers are commonly used architectures.",
        ],
    },
    "deep_learning": {
        "patterns": [
            "what is deep learning", "explain deep learning",
            "how does deep learning work", "deep learning vs machine learning",
            "neural networks in ai",
        ],
        "responses": [
            "Deep learning is a subset of machine learning that uses neural networks "
            "with many layers to learn complex patterns. It excels in tasks like image "
            "recognition, speech processing, and NLP. Architectures include CNNs "
            "(for images), RNNs/LSTMs (for sequences), and Transformers (for NLP).",
        ],
    },
    "reinforcement_learning": {
        "patterns": [
            "what is reinforcement learning", "explain reinforcement learning",
            "how does reinforcement learning work", "rl in ai",
        ],
        "responses": [
            "Reinforcement Learning (RL) trains agents to make decisions by rewarding "
            "desired behaviors and penalizing undesired ones. The agent learns a policy "
            "to maximize cumulative reward. Applications include game playing (AlphaGo), "
            "robotics, autonomous driving, and recommendation systems.",
        ],
    },
}

# ── Build the TF-IDF model ─────────────────────────────────────────────────

all_patterns = []
pattern_intents = []

for intent_name, intent_data in INTENTS.items():
    for pattern in intent_data["patterns"]:
        all_patterns.append(preprocess(pattern))
        pattern_intents.append(intent_name)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_patterns)

SIMILARITY_THRESHOLD = 0.15


def classify_intent(user_input):
    """Classify user input into the best-matching intent using cosine similarity."""
    processed = preprocess(user_input)
    user_vec = vectorizer.transform([processed])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    if best_score < SIMILARITY_THRESHOLD:
        return None, float(best_score)
    return pattern_intents[best_idx], float(best_score)


def get_response(user_input):
    """Get a chatbot response for the given user input."""
    intent, score = classify_intent(user_input)
    if intent is None:
        return {
            "response": "I'm not sure I understand that. Could you rephrase? "
                        "I can help with NLP concepts (tokenization, NER, transformers, etc.) "
                        "and AI applications (healthcare, finance, robotics, etc.).",
            "intent": "unknown",
            "confidence": round(score, 3),
        }
    response = random.choice(INTENTS[intent]["responses"])
    return {
        "response": response,
        "intent": intent,
        "confidence": round(score, 3),
    }


# ── Flask app ───────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please type a message.", "intent": "empty", "confidence": 0})
    result = get_response(user_message)
    return jsonify(result)


@app.route("/suggestions")
def suggestions():
    """Return a list of suggested questions for the UI."""
    return jsonify([
        "What is NLP?",
        "Explain tokenization",
        "How does sentiment analysis work?",
        "What is BERT?",
        "AI applications in healthcare",
        "What are word embeddings?",
        "How do transformers work?",
        "What is deep learning?",
    ])


if __name__ == "__main__":
    print("\n  NLP Chatbot Web UI")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
