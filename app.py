from flask import Flask,render_template,url_for,request
import pandas as pd
import re
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    sw = set(stopwords.words('english'))
    ps = nltk.PorterStemmer()

    data = pd.read_csv(r'data/SMSSpamCollection.tsv', sep='\t')
    data.columns = ['label', 'body_text']
    
    def clean_text(text):
        text = "".join([word.lower() for word in text if word not in string.punctuation])
        tokens = re.split('\W+', text)
        text = [ps.stem(word) for word in tokens if word not in sw]
        return text
    
    cv = CountVectorizer(analyzer=clean_text)
    X_counts = cv.fit_transform(data['body_text'])
    
    spam_model = open('model.pkl','rb')
    classifier = joblib.load(spam_model)

    if request.method == 'POST':
	    message = request.form['message']
	    data = [message]
	    vect = cv.transform(data).toarray()
	    my_prediction = classifier.predict(vect)
    return render_template('index.html', text = message, prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)