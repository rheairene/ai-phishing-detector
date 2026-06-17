# 🛡️ AI-Generated Phishing Email Detector

A machine learning tool that classifies emails as **Safe**, **Phishing**, or **AI-Generated Phishing** — with a live web interface where anyone can paste an email and get an instant result.

Built in 3 days by Rhea Irene Albert, age 18.

---

## Why This Exists

Phishing emails aren't new. But in 2026, AI-generated phishing emails are a growing and underdetected threat — they're polished, formally written, and bypass many traditional filters that look for bad grammar or obvious red flags.

Most public phishing datasets don't include this category at all. This project addresses that gap by:
- Training a classifier on 18,000+ real labelled emails
- Adding a custom hand-curated dataset of AI-generated phishing samples
- Building a live tool anyone can use to check suspicious emails


---

## Results

| Category | Precision | Recall | F1 Score |
|---|---|---|---|
| AI Phishing Email | 1.00 | 1.00 | 1.00 |
| Phishing Email | 0.94 | 0.98 | 0.96 |
| Safe Email | 0.99 | 0.96 | 0.97 |
| **Overall Accuracy** | | | **97%** |

---

## How It Works

**Dataset**
- Base: 18,650 emails from the [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) on Hugging Face (Safe Email / Phishing Email)
- Custom: 30 hand-crafted AI-generated phishing emails added as a third category

**Pipeline**
1. Text preprocessing — lowercasing, URL/email/link normalization, punctuation removal
2. TF-IDF vectorization (5,000 features, unigrams + bigrams)
3. Logistic Regression with `class_weight='balanced'` to handle class imbalance
4. Streamlit web app for live inference

**Why Logistic Regression?**
Fast, interpretable, and highly effective for text classification. At 97% accuracy on this dataset, more complex models aren't necessary — and interpretability matters in security tools.

---

## Project Structure

```
ai-phishing-detector/
│
├── app.py                        # Streamlit web app
├── train_model.py                # Full training pipeline
├── phishing_model.pkl            # Trained model
├── vectorizer.pkl                # TF-IDF vectorizer
├── label_encoder.pkl             # Label encoder
├── requirements.txt
├── runtime.txt    
└── README.md
```

---

## Run It Yourself

**Install dependencies:**
```bash
pip install streamlit scikit-learn pandas datasets
```

**Train the model:**
```bash
python train_model.py
```

**Launch the app:**
```bash
streamlit run app.py
```

---

## Key Observations

During development, three distinct patterns emerged across email categories:

**Safe emails** — conversational, context-specific, no calls to action, personal tone

**Human phishing emails** — urgent language, threats, poor grammar, suspicious URLs, generic greetings

**AI-generated phishing emails** — overly polished, excessively formal ("We hope this message finds you well"), vague calls to action ("[Verify My Account]"), no specific account details, generic sign-offs ("Warm regards, Customer Support Team")

The model learned to distinguish these patterns with 97% accuracy — and correctly classified all 6 AI phishing emails in the test set.

---

## Limitations & Future Work

- The AI phishing category has only 30 training examples — a larger, more diverse dataset would improve robustness
- The model is text-only — real phishing detection also uses metadata (sender domain, headers, link analysis)
- Future versions could use transformer-based models (BERT, RoBERTa) for better contextual understanding
- A browser extension would make this practically deployable

---

## About

**Rhea Irene Albert** — 18,. Self-taught in AI/ML and cybersecurity.

- IBM SkillsBuild Cybersecurity Certified (2026)
- AI internships at the Asian Institute of Technology, Thailand (2024)
- HubSpot Content Marketing Certified(2025)

📧 rheairene007@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/rheairene)
