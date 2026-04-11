import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv('smartphones.csv')
    df = df.fillna('')

    df['model'] = df['model'].str.lower().str.strip()
    df['brand_name'] = df['brand_name'].str.lower().str.strip()
    df['os'] = df['os'].str.lower().str.strip()
    df['processor_brand'] = df['processor_brand'].str.lower().str.strip()
    df['battery_capacity'] = pd.to_numeric(df['battery_capacity'], errors='coerce').fillna(0)
    df['ram_capacity'] = pd.to_numeric(df['ram_capacity'], errors='coerce').fillna(0)
    df['internal_memory'] = pd.to_numeric(df['internal_memory'], errors='coerce').fillna(0)
    df['refresh_rate'] = pd.to_numeric(df['refresh_rate'], errors='coerce').fillna(0)

    df['metadata'] = (
        df['brand_name'] + " " +
        df['os'] + " " +
        df['processor_brand'] + " " +
        df['battery_capacity'].astype(str) + " mAh " +
        df['ram_capacity'].astype(str) + " GB " +
        df['internal_memory'].astype(str) + " GB " +
        df['refresh_rate'].astype(str) + " Hz"
    )

    return df


df = load_data()

# ==============================
# TF-IDF + SIMILARITY
# ==============================
@st.cache_resource
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['metadata'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = compute_similarity(df)
indices = pd.Series(df.index, index=df['model']).drop_duplicates()

# ==============================
# RECOMMENDATION FUNCTION
# ==============================
def get_recommendations(model, cosine_sim=cosine_sim):
    model = model.lower().strip()

    if model not in indices:
        return None

    idx = indices[model]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    smart_indices = [i[0] for i in sim_scores]

    return df.iloc[smart_indices][['model', 'brand_name', 'os']]

# ==============================
# EVALUATION FUNCTION
# ==============================
def evaluate_system(df, recommendations_func, k=10, sample_size=50):
    sample_size = min(sample_size, len(df))
    test_samples = df.sample(sample_size, random_state=42)

    precision_scores = []

    for _, row in test_samples.iterrows():
        target_name = row['model']
        target_brand = row['brand_name']

        recommendations = recommendations_func(target_name)

        if recommendations is None:
            continue

        rec_top_k = recommendations.head(k)

        relevant_count = rec_top_k[
            rec_top_k['brand_name'] == target_brand
        ].shape[0]

        precision_at_k = relevant_count / k
        precision_scores.append(precision_at_k)

    return np.mean(precision_scores) if precision_scores else 0

# ==============================
# STREAMLIT UI
# ==============================
st.title("📱 Smartphone Recommendation System")

user_input = st.text_input("Enter Smartphone Name:")

if st.button("Get Recommendations"):
    results = get_recommendations(user_input)

    if results is None:
        st.error("Smartphone not found in dataset.")
    else:
        st.success(f"Recommendations for '{user_input}':")
        st.dataframe(results)

# ==============================
# EVALUATION CHART
# ==============================
st.subheader("📊 Precision@K Evaluation Chart")

if st.button("Generate Evaluation Chart"):
    k_values = list(range(1, 11))
    scores = []

    for k in k_values:
        score = evaluate_system(df, get_recommendations, k=k)
        scores.append(score)

    fig, ax = plt.subplots()
    ax.plot(k_values, scores, marker='o')
    ax.set_xlabel("K (Top Recommendations)")
    ax.set_ylabel("Precision@K")
    ax.set_title("Precision@K vs K")

    st.pyplot(fig)

    # show table
    result_df = pd.DataFrame({
        'K': k_values,
        'Precision': scores
    })
    st.dataframe(result_df)
