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
category_mapping = {'Books': 0, 'Clothing & Accessories': 1, 'Electronics': 2, 'Household': 3}

def data_cleaning(text):
    
    # Step 1: Remove non-alphanumeric characters
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Step 2: Remove extra spaces
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text.lower()

# web app
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = data_cleaning(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = ovr_knn.predict(input_features)[0]
        st.write(prediction_id)

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)



# python main
if __name__ == "__main__":
    main()