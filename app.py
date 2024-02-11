import streamlit as st
import PyPDF2 as pdf
from io import BytesIO
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import requests

st.set_page_config(layout="wide")

punctuation = punctuation + '\n' + '\t'
def summarizeText(text):

    # loading stopwords
    stop_words = list(STOP_WORDS)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    # tokenising the text
    tokens = [token.text for token in doc]

    
    # calculating the wordfrequency
    word_freq = {}
    for word in doc:
        if word.text.lower() not in stop_words:
            if word.text.lower() not in punctuation:
                if word.text not in word_freq.keys():
                    word_freq[word.text] = 1
                else:
                    word_freq[word.text] += 1

    # normalising the frequency
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word]/max_freq  


    # extracting the title
    title = " "
    for keys in word_freq.keys():
        # or word_freq[keys] == 0.5
        if word_freq[keys] == 1:
            title = keys

    # calculating sentence score
    sentence_tokens = [sent for sent in doc.sents]
    sentence_score = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_freq.keys():
                if sent not in sentence_score.keys():
                    sentence_score[sent] = word_freq[word.text.lower()]
                else:
                    sentence_score[sent] += word_freq[word.text.lower()]

    # extracting top 20% text having highest sentence score
    select_length = int(len(sentence_tokens)*0.2)
    summary = nlargest(select_length, sentence_score, key = sentence_score.get)
    dummy_summary = [word.text for word in summary]
    summary = ' '.join(dummy_summary)

    result = []
    result.append(title)
    result.append(summary)
    return result

# extracting text from pdf
def readPdfFile(input_file):
    # file = open(input_file.getvalue(),'rb')
    # pdf_reader = pdf.PdfReader(file)
    pdf_reader = pdf.PdfReader(BytesIO(input_file.getvalue()))
    text = ""
    for i in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[i].extract_text()

    result = summarizeText(text)
    return result

# Reading and summersing text
def readTextFile(input_text):
    return summarizeText(input_text)

# extracting text from pdf for Hugging face
def extractText(input_file):
    pdf_reader = pdf.PdfReader(BytesIO(input_file.getvalue()))
    text = ""
    for i in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[i].extract_text()
    return readTextUsingHuggingFacePDF(text)

# Summmerising text using hugging face
def readTextUsingHuggingFacePDF(text):

    API_URL = "https://api-inference.huggingface.co/models/Falconsai/text_summarization"
    headers = {"Authorization": f"Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
        
    output = query({
        "inputs": text
    })
    return output


# Designing the interface using streamlit
def main():
    choice = st.sidebar.selectbox("Select your choice", ["Upload Document","Paste Text","Using Hugging Face(document)","Using Hugging Face(text)"])

    if choice == "Upload Document":
        st.subheader("Summarize Document")
        input_file = st.file_uploader("Upload Your File", type=["pdf"])
        if input_file is not None:
            if st.button("Summarise Document"):
                result_text = readPdfFile(input_file)
                with st.container():
                    st.markdown("Title")
                    st.info(result_text[0])
                with st.container():
                    st.markdown("Summarized text") 
                    st.info(result_text[1])
                    
    elif choice == "Paste Text":
        st.subheader("Summarize Text")
        input_text = st.text_area("Enter Your Text")
        if st.button("Summarize text"):
            final_text = readTextFile(input_text)
            with st.container():
                st.markdown("Title")
                st.info(final_text[0])
            with st.container():
                st.markdown("Summarized Text")
                st.info(final_text[1])

    elif choice == "Using Hugging Face(document)":
        st.subheader("Summarize Document")
        input_file = st.file_uploader("Upload Your File", type=["pdf"])
        if input_file is not None:
            if st.button("Summarise Document"):
                result_text = extractText(input_file)
                with st.container():
                    st.markdown("Summarized text") 
                    st.info(result_text[0]['summary_text'])

    elif choice == "Using Hugging Face(text)":
        st.subheader("Summarize Text")
        input_text = st.text_area("Enter Your Text")
        if st.button("Summarize text"):
            final_text = readTextUsingHuggingFacePDF(input_text)
            with st.container():
                st.markdown("Summarized Text")
                st.info(final_text[0]['summary_text'])
                

if __name__ == "__main__":
    main()
