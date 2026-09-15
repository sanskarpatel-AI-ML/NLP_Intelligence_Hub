import streamlit as st
import joblib
import random

sentiment_model=joblib.load('sentiment_model.pkl')
sentiment_vectorizer=joblib.load('sentiment_vectorizer.pkl')

spam_model=joblib.load('spam_model.pkl')
spam_vectorizer=joblib.load('spam_vectorizer.pkl')

news_model = joblib.load("news_model.pkl")
news_vectorizer = joblib.load("news_vectorizer.pkl")

def result_card(icon, title, confidence=None, result_type="success"):

    if result_type == "success":
        bg = "#e8f5e9"
        border = "#43a047"
    else:
        bg = "#ffebee"
        border = "#e53935"

    confidence_text = ""

    if confidence is not None:
        confidence_text = f"""
<p style="margin: 8px 0 0 0; font-size: 16px;">
    Confidence: <b>{confidence:.2f}%</b>
</p>
"""

    st.markdown(
        f"""
<div style="
    background-color: {bg};
    border-left: 6px solid {border};
    padding: 20px;
    border-radius: 10px;
    margin-top: 15px;
">

<p style="
    margin: 0;
    font-size: 15px;
    font-weight: 600;
">
    📊 ANALYSIS RESULT
</p>

<h2 style="
    margin: 8px 0 0 0;
">
    {icon} {title}
</h2>

{confidence_text}

</div>
""",
        unsafe_allow_html=True
    )


def set_sentiment_example():
    st.session_state.ti1 = st.session_state.sentiment_example

def set_spam_example():
    spam_examples = [
        "Hey, are we still meeting for lunch today?",
        "Please send me the report when you get a chance.",
        "I've reached sch already. Let me know when you arrive.",
        "Congratulations! You have won a free prize. Call now to claim your reward.",
        "FREE MESSAGE! You have been selected for a special cash prize. Call now to claim."
    ]

    st.session_state.ti2 = random.choice(spam_examples)

def set_news_challenge():
    news_examples = [
        "Apple unveils a new AI-powered device with advanced features.",
        "Global markets rise as investors react to strong economic growth.",
        "The latest blockbuster film breaks box office records worldwide.",
        "Government announces new policies ahead of national elections.",
        "Microsoft invests billions in artificial intelligence research.",
        "Major banks report strong quarterly earnings amid market growth.",
        "Popular actor announces a new movie scheduled for release next year.",
        "Political leaders begin campaigning ahead of the upcoming election."
    ]

    st.session_state.news_input = random.choice(news_examples)

st.set_page_config(
    page_title="NLP Intelligence Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- SIDEBAR ----------
st.sidebar.image("NLP_image.png", use_container_width=True)

st.sidebar.markdown("---")

st.sidebar.subheader("📌 About the Project")
st.sidebar.write(
    "An interactive NLP application that uses "
    "machine-learning models to analyze and classify text."
)

st.sidebar.subheader("🤖 Available Models")
st.sidebar.write("💬 Sentiment Analysis")
st.sidebar.write("🚫 Spam Detection")
st.sidebar.write("📰 News Classification")

st.sidebar.markdown("---")

st.sidebar.subheader("🛠️ Technologies")
st.sidebar.write("Python • NLP • Machine Learning • Streamlit")
# ---------- MAIN HEADER ----------
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1f4e78, #2f75b5);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
">
    <h1 style="margin:0; font-size:40px;">
        🧠 NLP Intelligence Hub
    </h1>
    <p style="
        margin:10px 0 0 0;
        font-size:18px;
    ">
        Analyze and classify text using Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

tab1, tab2, tab3 = st.tabs([
    "💬  Sentiment Analysis",
    "🚫  Spam Detection",
    "📰  News Classification"
])

with tab1:

    st.subheader("💬 Sentiment Analysis")
    st.write("Understand the emotional tone of a piece of text.")

    # Default value
    if "ti1" not in st.session_state:
        st.session_state.ti1 = ""

    text = st.text_area(
        "Enter your message",
        placeholder="Type or paste your message here...",
        height=150,
        key="ti1"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        analyze = st.button(
            "🔍 Analyze Sentiment",
            key="sntb"
        )

    with col2:
        sentiment_examples = [
            "Select an example...",
            "I was disappointed with the product and would not recommend it.",
            "The service was excellent and the quality exceeded my expectations.",
            "The service was poor and the overall experience was frustrating.",
            "I really enjoyed the experience and would definitely recommend it.",
            "This product is amazing and works exactly as expected."
        ]

        st.selectbox(
            "📝 Select an Example",
            sentiment_examples,
            key="sentiment_example",
            on_change=set_sentiment_example
        )

    if analyze:

        if text.strip() == "":
            st.warning("⚠️ Please enter some text first.")

        else:

            text_vector = sentiment_vectorizer.transform([text])
            pred = sentiment_model.predict(text_vector)

            confidence = sentiment_model.predict_proba(text_vector)[0]

            st.markdown("---")
            st.markdown("### 📊 Analysis Result")

            if pred[0] == 0:

                result_card(
                    "😞",
                    "NEGATIVE",
                    confidence[0] * 100,
                    "error"
                )

            else:

                result_card(
                    "😊",
                    "POSITIVE",
                    confidence[1] * 100,
                    "success"
                )

with tab2:

    st.subheader("🚫 Spam Detection")
    st.write("Determine whether a message is spam or legitimate.")

    if "ti2" not in st.session_state:
        st.session_state.ti2 = ""

    text = st.text_area(
        "Enter your message",
        placeholder="Type or paste a message here...",
        height=150,
        key="ti2"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        check_spam = st.button(
            "🔍 Check Message",
            key="spmb"
        )

    with col2:
        st.button(
            "💡 Use Example",
            key="spam_example",
            on_click=set_spam_example
        )

    if check_spam:

        if text.strip() == "":
            st.warning("⚠️ Please enter a message first.")

        else:

            text_vector = spam_vectorizer.transform([text])
            pred = spam_model.predict(text_vector)

            confidence = spam_model.predict_proba(text_vector)[0]

            st.markdown("---")
            st.markdown("### 📊 Analysis Result")

            if pred[0] == "spam":

                result_card(
                    "🚫",
                    "SPAM",
                    confidence[1] * 100,
                    "error"
                )
            else:

                result_card(
                    "✅",
                    "NOT SPAM",
                    confidence[0] * 100,
                    "success"
                )

with tab3:

    st.subheader("📰 News Classification")
    st.write("Classify a news headline into its appropriate category.")

    if "news_input" not in st.session_state:
        st.session_state.news_input = ""

    headline = st.text_area(
        "Enter your news headline",
        placeholder="Type or paste a news headline here...",
        height=120,
        key="news_input"
    )

    st.write("### 🎯 What do you think this news belongs to?")

    guess = st.radio(
        "Your Guess",
        ["Tech", "Business", "Entertainment", "Politics"],
        horizontal=True
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        classify_news = st.button(
            "🔍 Classify News",
            key="newsb"
        )

    with col2:
        st.button(
            "🎯 New Challenge",
            key="news_challenge",
            on_click=set_news_challenge
        )

    if classify_news:

        if headline.strip() == "":
            st.warning("⚠️ Please enter a news headline first.")

        else:

            headline_vector = news_vectorizer.transform([headline])
            prediction = news_model.predict(headline_vector)

            st.markdown("---")
            st.markdown("### 📊 Classification Result")

            result_card(
                "📰",
                f"{prediction[0]}",
                result_type="success"
            )     

st.markdown("---")

st.markdown(
f"""
<div style="text-align: center; padding: 20px 0; color: #6b7280; font-size: 14px;">
<div>
Built by <b>Sanskar Patel</b>
</div>

<div style="margin-top: 6px;">
Python • Machine Learning • NLP • Streamlit
</div>

<div style="margin-top: 6px; font-size: 12px;">
© 2026 Sanskar Patel
</div>
</div>
""",
unsafe_allow_html=True
)   
    