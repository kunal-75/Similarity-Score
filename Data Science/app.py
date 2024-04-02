from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json  # Retrieve JSON data
    text1 = data['text1']
    text2 = data['text2']

    print(text1+" "+text2)

    def transform_text(text):
        # Lowercase
        text = text.lower()
        # Tokenization
        text = nltk.word_tokenize(text)
        # Removing Special Characters
        y = []
        for i in text:
            if (i.isalnum()):
                y.append(i)
        # Removing Stop Words & Punctuations
        text = y[:]
        y.clear()
        for i in text:
            if (i not in stopwords.words('english') and i not in string.punctuation):
                y.append(i)
        # Stemming
        text = y[:]
        y.clear()
        for i in text:
            y.append(ps.stem(i))
        return " ".join(y)

    text1 = transform_text(text1)
    text2 = transform_text(text2)

    def cosine_sim(text1, text2):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf1 = vectorizer.fit_transform([text1, text2])
        cosine_sim = cosine_similarity(tfidf1[0], tfidf1[1])
        return cosine_sim[0][0]

    similarity_score = cosine_sim(text1, text2)
    return jsonify({"similarity score": similarity_score})

if __name__ == '__main__':
    app.run(debug=True)
