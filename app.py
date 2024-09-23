import streamlit as st
from langchain.chains import create_history_aware_retriever,create_retrieval_chain #history aware retriever to add history along with history prompt
from langchain.chains.combine_documents import create_stuff_documents_chain #combine all the documents and send it to the context
from langchain_chroma import  Chroma
from langchain_community.chat_message_histories import  ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder #Message place holder to define session key and what kind of message we are storing in the session key
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

#load the environment variables and define embedding layers
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDF's and chat with the content")

#Enter the groq api key
api_key = st.text_input("Enter your Groq api key:",type="password")


#check if api key is provided
if api_key:

    #create llm model
    llm = ChatGroq(groq_api_key=api_key,model_name="gemma2-9b-it")
    
    #define session id
    session_id = st.text_input("session ID",value="default_session")

    #check if store is present in the memory (statefully manage chat history)
    if "store" not in st.session_state:
        st.session_state.store={}
    
    uploaded_files = st.file_uploader("Choose a PDF file",type="pdf",accept_multiple_files=True)

    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            #store the file locally
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader = PyPDFLoader(temp_pdf)
            docs=loader.load()
            documents.extend(docs)

        #split the data into text chunks , apply embeddings and store them in vectorstoredb
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits,embedding=embedding)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history"
            "formulate a standalone question which can be understood"
            "without chat history.Do not answer the question,"
            "just formulate it if needed and otherwise return it as is"
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user","{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        #define Q&A prompt
        system_prompt = (
            "you are an assistant for question answering tasks"
            "use the following pieces of retrieved context to answer"
            "the question.If you don't know the answer say that you don't know"
            "Use three sentences maximum and keep the answer precise and concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user","{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        #get session history for a particular session id
        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your Question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversation_rag_chain.invoke(
                {"input":user_input},
                config = {
                    "configurable":{"session_id":session_id}
                }
            )

            st.write(st.session_state.store)
            st.write("Assistant: ",response['answer'])
            st.write("Chat History: ",session_history.messages)

else:
    st.warning("Please enter Groq api key")
