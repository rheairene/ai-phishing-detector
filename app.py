import streamlit as st
import pickle
import re

with open('phishing_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', 'clicklink', text)
    text = re.sub(r'http\S+|www\S+', 'urllink', text)
    text = re.sub(r'\S+@\S+', 'emailaddr', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.set_page_config(page_title='AI Phishing Detector', page_icon='🛡️')
st.title('🛡️ AI-Generated Phishing Email Detector')
st.markdown('Paste any email below to check if it is **safe**, a **phishing attempt**, or an **AI-generated phishing email**.')
st.markdown('---')

email_input = st.text_area('📧 Paste email content here:', height=250, placeholder='Paste the full email text here...')

if st.button('🔍 Analyze Email'):
    if email_input.strip() == '':
        st.warning('Please paste an email to analyze.')
    else:
        cleaned = clean_text(email_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        label = le.inverse_transform([prediction])[0]
        confidence = round(max(probabilities) * 100, 2)
        st.markdown('---')
        if label == 'Safe Email':
            st.success(f'✅ SAFE EMAIL — Confidence: {confidence}%')
            st.markdown('This email appears to be legitimate.')
        elif label == 'Phishing Email':
            st.error(f'🚨 PHISHING EMAIL — Confidence: {confidence}%')
            st.markdown('This email shows patterns consistent with human-written phishing attempts.')
        else:
            st.error(f'🤖 AI-GENERATED PHISHING EMAIL — Confidence: {confidence}%')
            st.markdown('This email shows patterns consistent with AI-generated phishing.')
        st.markdown('---')
        st.markdown('**Confidence Breakdown:**')
        for i, class_name in enumerate(le.classes_):
            st.write(f'{class_name}: {round(probabilities[i] * 100, 2)}%')
            st.progress(float(probabilities[i]))

st.markdown('---')
st.caption('Built by Rhea Irene Albert | IBM SkillsBuild Cybersecurity | 2026')
