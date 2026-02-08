import streamlit as st
from PyPDF2 import PdfReader

st.set_page_config(page_title='AI Twin')
st.title('AI personal Knowledge Twin')

st.write("uplod your notes and asl anything about it")
uploaded_file=st.file_uploader("Upload a pdf file",type=['pdf'])

question = st.text_input(
    "Ask a question from your knowledge"
)
if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    st.subheader("Extracted Text (Preview)")
    st.write(text[:1000])

if uploaded_file and question:
    st.info("Next step: connect AI to answer this question")

