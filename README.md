# 🤖 AI-BASED ELECTRONICS RECOMMENDATION SYSTEM (CHOOSE AI)

### Description of an AI-Driven Electronics Recommendation System Using Machine Learning and Web Technologies in Python

In the modern digital marketplace, users are often overwhelmed by the vast number of electronic products available online. Intelligent recommendation systems play a crucial role in assisting users by analyzing preferences and suggesting relevant products. CHOOSE AI is an AI-driven electronics recommendation system that leverages machine learning and natural language processing techniques to provide personalized product recommendations. The system simulates a conversational AI assistant, guiding users through the product selection process in an interactive and user-friendly manner.

### Conceptual Framework

The recommendation process in CHOOSE AI is composed of multiple stages, each utilizing specific tools and methodologies:

### **Environment Setup and Library Installation:**
The system is developed using Python and relies on essential libraries such as `Streamlit` for the web interface, `Pandas` for data manipulation, and `Scikit-learn` for implementing machine learning algorithms. These libraries collectively enable data processing, model execution, and interactive visualization.

### **Data Collection and Preparation:**
A structured dataset containing electronic product details such as product name, brand, category, price, rating, and features is used. Textual attributes are combined and preprocessed to form a unified representation suitable for machine learning analysis.

### **Machine Learning-Based Feature Extraction:**
The project employs TF-IDF (Term Frequency–Inverse Document Frequency) vectorization to convert textual product information into numerical vectors. This technique captures the importance of keywords within product descriptions, enabling meaningful similarity comparisons.

```python
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(product_text)
```

### **Similarity Computation and Recommendation:**
Cosine similarity is used to measure the closeness between user queries and product vectors. Products with the highest similarity scores are identified as the most relevant recommendations, along with a list of similar alternatives.

```python
similarity = cosine_similarity(query_vector, tfidf_matrix)
```

### **Conversational User Interaction:**
The system follows a chat-style interaction flow, where the AI assistant gathers user preferences such as brand, budget, and usage category. Based on these inputs, the system refines recommendations to closely match user requirements.

### **Result Presentation and External Integration:**
Recommended products are displayed along with dynamically generated Amazon and Flipkart purchase links, enabling users to explore real-world buying options. If no suitable products are found, the system provides feedback and allows users to restart the recommendation process.

### **Applications and Extensions**

CHOOSE AI can be extended to include advanced recommendation techniques such as collaborative filtering, deep learning-based embeddings, and real-time API integration. The system can also be deployed as a web service for e-commerce platforms or enhanced with large language models for richer explanations.

### Conclusion

CHOOSE AI demonstrates the practical application of machine learning and natural language processing in building an intelligent recommendation system. By combining content-based filtering with an interactive user interface, the project offers an effective and scalable solution for personalized electronics recommendation in real-world scenarios.

## OUTPUT

#### Homepage

![Image](https://github.com/user-attachments/assets/6ac21a07-22f2-4592-a522-57dd972dafb4)

#### Recommended Product

![Image](https://github.com/user-attachments/assets/46c533e6-00b7-42da-8398-92ac58c3ed02)
