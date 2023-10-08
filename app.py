import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from htmlTemplates import css, bot_template, user_template

# Langchain imports
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = "".join(page.extract_text() for pdf in pdf_docs 
                   for page in PdfReader(pdf).pages)
    return text


# Function to split the extracted text into chunks
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, 
        chunk_overlap=200, length_function=len)
    
    return splitter.split_text(text)


# Function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# Function to initialize the conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory)


# Function to handle user input and display the conversation
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    templates = [user_template, bot_template]
    for i, message in enumerate(st.session_state.chat_history):
        st.write(templates[i % 2].replace(
            "{{MSG}}", message.content), unsafe_allow_html=True)


# Main function to drive the Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.session_state.setdefault("conversation", None)
    st.session_state.setdefault("chat_history", None)

    st.header("Chat with PDF ")
    user_question = st.text_input("Ask questions about your PDFs:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", 
            accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)


# Run the Streamlit app
if __name__ == '__main__':
    main()
