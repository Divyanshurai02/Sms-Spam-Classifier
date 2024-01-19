import streamlit as st
import pickle
import nltk
nltk.download('wordnet')
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load pre-trained model and vectorizer
ps = PorterStemmer()
tf = pickle.load(open('vectorizerr.pkl', 'rb'))
model = pickle.load(open('modell.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Streamlit UI
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱", layout="wide")

# Add custom inline styles

# Header
st.title("📱SMS Spam Classifier 🚫")

# Description
st.write(
    "Welcome to the SMS Spam Classifier! Enter a message in the text area below, and the model will predict whether it's spam or not."
)

# Input text area
input_sms = st.text_area("Please enter the message")

# preprocess
transform_sms = transform_text(input_sms)
vector_input = tf.transform([transform_sms])
result = model.predict(vector_input)[0]

# Predict button
if st.button("Predict",type='primary'):
    if len(input_sms)>0:
        if result == 1:
            st.error('Alert! This message is likely spam.', icon="🚨")

        else:
            st.success('This message is safe and not spam! ✅')
    else:
        st.warning("No Message⚠️ found to Predict,Please Enter!")

