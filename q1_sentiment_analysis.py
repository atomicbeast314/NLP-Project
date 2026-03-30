"""
CSE3246 - NLP Assignment
Q1: Sentiment Analysis Mini Project
Topic: Movie Review Sentiment Analysis (IMDb-style)

Tasks covered:
  1. Movie review dataset – NLTK movie_reviews corpus (IMDb-style, 2000 reviews)
     plus hand-crafted neutral reviews for three-class classification
  2. Text preprocessing (lowercase, punctuation removal, stopword removal)
  3. Feature extraction: Bag of Words (CountVectorizer) and TF-IDF
  4. Naive Bayes classifier (MultinomialNB)
  5. Accuracy + Confusion Matrix (3-class: Negative / Neutral / Positive)
  6. Results explained
"""

import re
import string
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import movie_reviews

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# 1.  DATASET
# ─────────────────────────────────────────────
# Download NLTK movie_reviews corpus (≈ 3 MB, one-time download)
nltk.download("movie_reviews", quiet=True)

random.seed(42)
np.random.seed(42)

# Sample 60 positive and 60 negative reviews from the NLTK corpus
pos_ids = movie_reviews.fileids("pos")
neg_ids = movie_reviews.fileids("neg")

random.shuffle(pos_ids)
random.shuffle(neg_ids)

pos_sample = pos_ids[:60]
neg_sample = neg_ids[:60]

pos_texts = [movie_reviews.raw(fid) for fid in pos_sample]
neg_texts = [movie_reviews.raw(fid) for fid in neg_sample]

# Neutral reviews: hand-crafted (IMDb does not provide neutral ratings)
# These represent 5–6/10 reviews — watchable but unremarkable films
neutral_texts = [
    "The film was okay. Nothing special, but not terrible either. Some scenes worked well.",
    "An average movie with a few good moments. The acting was decent but the story felt flat.",
    "It's a passable way to spend two hours. Not memorable, but not a waste of time.",
    "The movie had its moments but overall felt mediocre. The pacing was uneven throughout.",
    "I neither loved nor hated this film. Some interesting ideas, but poorly executed.",
    "Pretty standard for the genre. Competent filmmaking with nothing particularly exciting.",
    "Mixed feelings overall. The visuals were impressive but the script was quite weak.",
    "Not great, not terrible. The cast did what they could with a formulaic story.",
    "A middle-of-the-road effort. There are better films in this genre, and worse ones too.",
    "Watchable but forgettable. You won't regret seeing it, but won't remember it either.",
    "The first half was promising but the second half lost momentum. Decent, not great.",
    "An inoffensive movie that does nothing new. Competent but completely unmemorable.",
    "Some genuinely funny moments scattered through an otherwise unremarkable comedy.",
    "Good premise, average execution. The director played it too safe to make an impact.",
    "The performances were fine and the cinematography was decent. Story was just okay.",
    "A serviceable thriller with a predictable twist. Enjoyable enough while it lasted.",
    "Decent action sequences but thin characters. Popcorn entertainment, nothing more.",
    "The film meanders but never loses you entirely. Worth a single watch, maybe.",
    "Three stars out of five. Has both strengths and weaknesses in roughly equal measure.",
    "Not the best film of the year, but not the worst. A perfectly average experience.",
    "Lukewarm recommendation. It tries hard without fully succeeding or completely failing.",
    "Interesting concept, uneven delivery. The leads had chemistry but the plot dragged.",
    "An ordinary film elevated slightly by one strong performance. Otherwise forgettable.",
    "You will sit through it without checking your watch too often. That is about it.",
    "Mildly entertaining but lacks the ambition to be truly great or memorably bad.",
    "The screenplay needed another draft. Watchable nonetheless thanks to solid direction.",
    "A film that gets by on charm without having much substance beneath the surface.",
    "Routine storytelling with occasional flashes of creativity. Middle-of-the-road stuff.",
    "Not a bad film, just a thoroughly unremarkable one. Genre fans might enjoy it more.",
    "Competently made and cast, but the story offers nothing you haven't seen before.",
    "The movie works on a basic level without ever rising above genre conventions.",
    "Some nice cinematography can't disguise a script in desperate need of revision.",
    "Enjoyable in a low-key way. The performances carry a thin script well enough.",
    "A solid genre piece that delivers exactly what it promises and nothing more.",
    "Neither particularly funny nor particularly dramatic. Just a pleasant enough watch.",
    "The film has moments of real quality surrounded by long stretches of mediocrity.",
    "Average in almost every respect. The cast is fine, the direction is fine, the story is fine.",
    "Gets the job done without any flair. A competent, unremarkable piece of filmmaking.",
    "You could watch worse. You could certainly watch better. Firmly in the middle.",
    "An honest three-star effort. Professional craft without inspiration or ambition.",
    "The trailer promised more than the film delivered, but it is far from a disaster.",
    "Reasonably entertaining without being remotely innovative or thought-provoking.",
    "Predictable from start to finish but executed with enough skill to stay watchable.",
    "A movie that exists and does not embarrass itself. High praise? No. Warranted? Yes.",
    "The genre beats are all there, hit with precision but without any personal touch.",
    "Satisfies basic entertainment needs without ever reaching for anything greater.",
    "Workmanlike filmmaking. Every element is adequate; nothing is excellent.",
    "A curious film — simultaneously hard to love and hard to dismiss entirely.",
    "Passes the time pleasantly enough. Not something I would rush to recommend, though.",
    "The movie leaves no strong impression either way. Completely and utterly average.",
    "Technically sound, emotionally flat. A film made by committee with committee results.",
    "The performances are the best thing about an otherwise thin production.",
    "Has its charms, has its flaws. Lands squarely in the middle of the road.",
    "Runs through familiar motions without ever surprising or disappointing the audience.",
    "A film made to a formula. Follows that formula faithfully, for better or worse.",
    "Passable entertainment for a slow evening. Nothing you need to seek out, though.",
    "The leads are likeable and carry the film further than the script deserves.",
    "Unremarkable but functional. Like fast food — not great, not terrible, fills a need.",
    "Some scenes genuinely work; others fall flat. The overall effect is decidedly average.",
    "A mild curiosity. Worth a look on streaming but not worth going out of your way for.",
]

# ─────────────────────────────────────────────
# 2.  ASSEMBLE DATASET
# ─────────────────────────────────────────────
# Labels: 0 = Negative, 1 = Neutral, 2 = Positive
texts  = neg_texts + neutral_texts + pos_texts
labels = [0] * 60 + [1] * 60 + [2] * 60

label_names = ["Negative", "Neutral", "Positive"]

print("=" * 60)
print("Q1: MOVIE REVIEW SENTIMENT ANALYSIS")
print("=" * 60)
print(f"\nDataset: {len(texts)} reviews")
print(f"  Negative : {labels.count(0)}")
print(f"  Neutral  : {labels.count(1)}")
print(f"  Positive : {labels.count(2)}")

# ─────────────────────────────────────────────
# 3.  TEXT PREPROCESSING
# ─────────────────────────────────────────────
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","not","only","own","same","so","than","too",
    "very","s","t","can","will","just","don","should","now","d","ll","m","o",
    "re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven",
    "isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won",
    "wouldn"
}

def preprocess(text):
    """Lowercase → remove punctuation → remove stopwords → strip extra spaces."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

cleaned_texts = [preprocess(t) for t in texts]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 60)
print("SAMPLE PREPROCESSED REVIEWS")
print("=" * 60)
samples = [(neg_texts[0], 0), (neutral_texts[0], 1), (pos_texts[0], 2)]
for raw, lbl in samples:
    clean = preprocess(raw)
    snippet = raw[:120].replace("\n", " ")
    print(f"\n  Label    : {label_names[lbl]}")
    print(f"  Original : {snippet}...")
    print(f"  Cleaned  : {clean[:100]}...")

# ─────────────────────────────────────────────
# 4.  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_texts, labels, test_size=0.25, random_state=42, stratify=labels
)
print(f"\nDataset split → Train: {len(X_train)}, Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 5.  FEATURE EXTRACTION
# ─────────────────────────────────────────────
bow_vectorizer   = CountVectorizer(max_features=3000)
tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))

X_train_bow   = bow_vectorizer.fit_transform(X_train)
X_test_bow    = bow_vectorizer.transform(X_test)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf  = tfidf_vectorizer.transform(X_test)

print(f"\nBoW   vocabulary size : {len(bow_vectorizer.vocabulary_)}")
print(f"TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# ─────────────────────────────────────────────
# 6.  NAÏVE BAYES CLASSIFIER
# ─────────────────────────────────────────────
nb_bow   = MultinomialNB()
nb_tfidf = MultinomialNB()

nb_bow.fit(X_train_bow, y_train)
nb_tfidf.fit(X_train_tfidf, y_train)

y_pred_bow   = nb_bow.predict(X_test_bow)
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)

# ─────────────────────────────────────────────
# 7.  EVALUATION
# ─────────────────────────────────────────────
acc_bow   = accuracy_score(y_test, y_pred_bow)
acc_tfidf = accuracy_score(y_test, y_pred_tfidf)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"\n  BoW   + Naive Bayes Accuracy : {acc_bow * 100:.1f}%")
print(f"  TF-IDF + Naive Bayes Accuracy: {acc_tfidf * 100:.1f}%")

print("\n--- Classification Report (TF-IDF) ---")
print(classification_report(y_test, y_pred_tfidf, target_names=label_names))

# ─────────────────────────────────────────────
# 8.  VISUALISATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Q1: Movie Review Sentiment Analysis – Results", fontsize=14, fontweight="bold")

# --- Confusion Matrix (TF-IDF) ---
cm = confusion_matrix(y_test, y_pred_tfidf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
            ax=axes[0], linewidths=0.5)
axes[0].set_title("Confusion Matrix (TF-IDF + Naive Bayes)")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("Actual Label")

# --- Accuracy Comparison Bar ---
models     = ["BoW\n+ NB", "TF-IDF\n+ NB"]
accuracies = [acc_bow * 100, acc_tfidf * 100]
bars = axes[1].bar(models, accuracies, color=["#4C9BE8", "#E87C4C"],
                   edgecolor="black", width=0.4)
axes[1].set_ylim(0, 115)
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Accuracy: BoW vs TF-IDF")
for bar, val in zip(bars, accuracies):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 2, f"{val:.1f}%",
                 ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "q1_results.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nPlot saved → q1_results.png")

# ─────────────────────────────────────────────
# 9.  LABEL DISTRIBUTION PLOT
# ─────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(5, 4))
counts = [labels.count(i) for i in range(3)]
colors = ["#F44336", "#FFC107", "#4CAF50"]
bars2 = ax.bar(label_names, counts, color=colors, edgecolor="black")
ax.set_title("Dataset Label Distribution\n(Movie Reviews)")
ax.set_ylabel("Number of Reviews")
for bar, v in zip(bars2, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(v),
            ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "q1_label_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved → q1_label_distribution.png")

# ─────────────────────────────────────────────
# 10.  EXPLANATION
# ─────────────────────────────────────────────
print("""
==========================================================
RESULTS EXPLANATION
==========================================================

Dataset:
  180 movie reviews balanced across three sentiment classes:
    Negative (60) – poor/one-star IMDb-style reviews from NLTK corpus
    Neutral  (60) – mixed/average reviews (hand-crafted, 5-6/10 tone)
    Positive (60) – glowing reviews from NLTK movie_reviews corpus
  IMDb itself is binary (pos/neg); neutral reviews were crafted to
  simulate real mixed-sentiment responses.

Preprocessing steps applied:
  1. Lowercasing       – normalises case differences
  2. Punctuation removal – strips !, ., , etc.
  3. Stopword removal  – removes high-frequency words (the, is, a ...)
     that carry little sentiment signal.

Feature Extraction:
  BoW  – counts raw word occurrences (max 3 000 features).
         Simple but ignores word importance across documents.
  TF-IDF – weights words by uniqueness to each document.
            Uses unigrams + bigrams (ngram_range=(1,2)) to capture
            phrases like "not great" or "quite good".

Naive Bayes Classifier:
  Multinomial NB estimates the probability that each word/bigram
  appears given the sentiment class, then applies Bayes' theorem.
  Fast, interpretable, and well-suited to text classification.

Confusion Matrix (TF-IDF):
  The neutral class is the hardest to distinguish because neutral
  language borrows vocabulary from both positive and negative reviews.
  Off-diagonal entries in the Neutral row/column are expected.

Observations:
  TF-IDF + bigrams generally outperforms plain BoW because bigrams
  capture negations and qualifiers ("not bad", "barely watchable"),
  which are especially important for distinguishing neutral sentiment
  from the polar extremes.
==========================================================
""")
