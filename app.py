import streamlit as st
import pandas as pd
import random
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="CHOOSE AI",
    layout="wide"
)

def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-repeat: no-repeat;
            background-position: center;
            background-size: 45%;
            background-attachment: fixed;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            height: 100%;
            width: 100%;
            background: rgba(255, 255, 255, 0.88);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("choose_ai_logo.png")

st.title("🤖 AI Electronics Shopping Assistant")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("electronics.csv")

df = load_data()

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def amazon_buy_link(name):
    return "https://www.amazon.in/s?k=" + name.replace(" ", "+")

def flipkart_buy_link(name):
    return "https://www.flipkart.com/search?q=" + name.replace(" ", "+")

def estimate_prices(price):
    return int(price * 0.92), int(price * 0.97)

def llm_explanation(user_pref, product):
    return f"""
### 🤖 AI Explanation

Based on your inputs:
- **Product:** {user_pref['query']}
- **Brand:** {user_pref['brand']}
- **Category:** {user_pref['category']}
- **Budget:** ₹{user_pref['budget'][0]} – ₹{user_pref['budget'][1]}

The product **{product['name']}** was selected because:
- It perfectly matches your **brand and category**
- It falls within your **budget**
- Its features closely align with your intent using NLP similarity

**Model Used:** TF-IDF + Cosine Similarity  
**Confidence:** High
"""

# -------------------------------------------------
# Session State (Chat Memory)
# -------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 0

if "user_pref" not in st.session_state:
    st.session_state.user_pref = {}

# -------------------------------------------------
# CHAT FLOW
# -------------------------------------------------

# STEP 0 – Greeting
if st.session_state.step == 0:
    st.chat_message("assistant").write(
        "👋 Hi! I’m your AI Electronics Assistant.\n\nWhat product are you looking for today?"
    )
    st.write("")  # spacing
    st.write("")  # spacing

    query = st.chat_input("Type a product name...")

    if query:
        st.session_state.user_pref["query"] = query
        st.session_state.step = 1
        st.rerun()

# STEP 1 – Brand
elif st.session_state.step == 1:
    st.chat_message("assistant").write("Which brand do you prefer?")
    brand = st.selectbox("Select Brand", sorted(df["brand"].unique()))

    if st.button("Next"):
        st.session_state.user_pref["brand"] = brand
        st.session_state.step = 2
        st.rerun()

# STEP 2 – Category
elif st.session_state.step == 2:
    st.chat_message("assistant").write("What category suits you best?")
    category = st.selectbox(
        "Category",
        sorted(df["category"].unique())
    )

    if st.button("Next"):
        st.session_state.user_pref["category"] = category
        st.session_state.step = 3
        st.rerun()

# STEP 3 – Budget
elif st.session_state.step == 3:
    st.chat_message("assistant").write("Select your budget range 💰")

    min_price = int(df["price"].min())
    max_price = int(df["price"].max())

    budget = st.slider(
        "Budget (₹)",
        min_price,
        max_price,
        (min_price, max_price)
    )

    if st.button("Get Recommendations"):
        st.session_state.user_pref["budget"] = budget
        st.session_state.step = 4
        st.rerun()

# STEP 4 – RECOMMENDATION
elif st.session_state.step == 4:

    prefs = st.session_state.user_pref

    # Filter dataset strictly
    filtered = df[
        (df["brand"] == prefs["brand"]) &
        (df["category"] == prefs["category"]) &
        (df["price"].between(prefs["budget"][0], prefs["budget"][1]))
    ]

    if filtered.empty:
        st.warning("❌ No products match your exact preferences.")

        st.write("")  # spacing

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Start Over"):
                st.session_state.step = 0
                st.session_state.user_pref = {}
                st.rerun()
        st.stop()

    # NLP similarity
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(
        filtered["name"] + " " + filtered["features"]
    )

    query_vec = tfidf.transform([prefs["query"]])
    similarity = cosine_similarity(query_vec, tfidf_matrix)[0]

    filtered = filtered.assign(score=similarity).sort_values(
        "score", ascending=False
    )

    matched_product = filtered.iloc[0]
    similar_products = filtered.iloc[1:4]

    # -------------------------------------------------
    # Matched Product
    # -------------------------------------------------
    st.chat_message("assistant").write("✅ **Here is the best match for you:**")

    st.subheader(matched_product["name"])
    st.write(f"**Brand:** {matched_product['brand']}")
    st.write(f"**Category:** {matched_product['category']}")
    st.write(f"**Rating:** ⭐ {matched_product['rating']}")

    st.write("")  # spacing

    fk_price, amz_price = estimate_prices(matched_product["price"])

    st.write("### 💰 Price Comparison")
    st.write(f"🛒 Flipkart: ₹{fk_price}")
    st.write(f"🛍 Amazon: ₹{amz_price}")

    col1, col2, col3, col4, col5 = st.columns([2, 1, 0.2, 1, 2],gap="small")

    with col1:
        st.link_button(
            "🛒 Flipkart",
            flipkart_buy_link(matched_product["name"]),
            use_container_width=False
        )

    with col2:
        st.link_button(
            "🛍 Amazon",
            amazon_buy_link(matched_product["name"]),
            use_container_width=False
        )

    st.markdown("---")

    # -------------------------------------------------
    # LLM Explanation
    # -------------------------------------------------
    st.markdown(
        llm_explanation(prefs, matched_product)
    )
    st.markdown("---")

    # -------------------------------------------------
    # Similar Products (Cards, NOT TABLE)
    # -------------------------------------------------
    st.subheader("📌 Similar Products You May Like")
    st.write("")  # spacing
    
    if similar_products.empty:
        st.info(
            "😔 Sorry! There are no similar products available as per your requirement."
        )
    else:
        for i, row in similar_products.iterrows():
            st.markdown(f"### 🔹 {row['name']}")
            st.write(f"**Price:** ₹{row['price']}")
            st.write(f"**Rating:** ⭐ {row['rating']}")
    
            col1, col2 = st.columns(2)
            col1.link_button("Buy on Flipkart", flipkart_buy_link(row["name"]))
            col2.link_button("Buy on Amazon", amazon_buy_link(row["name"]))
    
            st.divider()
        st.success("🎉 Hope you found what you were looking for!")
