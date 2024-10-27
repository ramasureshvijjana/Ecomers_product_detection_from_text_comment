import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

#loading models
ovr_knn = pickle.load(open('results/ovr_knn.pkl','rb'))
tfidfd = pickle.load(open('results/tfidf.pkl','rb'))

# Map category ID to category name
category_mapping = {0: 'Books', 1: 'Clothing & Accessories', 2: 'Electronics', 3:'Household'}

def data_cleaning(text):
    
    # Step 1: Remove non-alphanumeric characters
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Step 2: Remove extra spaces
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text.lower()

# web app
def main():
    st.title("Ecomers Product Detection From Text Comment App")
    comment_text = st.text_input("Enter Product Comment", "")
    
    # uploaded_file = st.file_uploader('Upload Comment as text file', type=['txt','pdf'])

    # if uploaded_file is not None:
    #     try:
    #         resume_bytes = uploaded_file.read()
    #         resume_text = resume_bytes.decode('utf-8')
    #     except UnicodeDecodeError:
    #         # If UTF-8 decoding fails, try decoding with 'latin-1'
    #         resume_text = resume_bytes.decode('latin-1')

    cleaned_comment_text = data_cleaning(comment_text)
    input_features = tfidfd.transform([cleaned_comment_text])
    prediction_id = ovr_knn.predict(input_features)[0]

    category_name = category_mapping.get(prediction_id, "Unknown")

    st.write("Predicted Category:", category_name)



# python main
if __name__ == "__main__":
    main()