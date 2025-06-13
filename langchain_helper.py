from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

llm=ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key=os.environ['api_key'], temperature=0.1)

embedding_model=HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path="faiss_index"

def create_vector_db():
    loader=CSVLoader(file_path='friends_faqs.csv', source_column="Question", autodetect_encoding=True)
    data=loader.load()

    vectordb=FAISS.from_documents(documents=data, embedding=embedding_model)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    vectordb=FAISS.load_local(vectordb_file_path, embedding_model, allow_dangerous_deserialization=True)
    retriever=vectordb.as_retriever(score_threshold=0.7)

    prompt_template="""Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "Answer" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT=PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain=RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True)

    return chain

if __name__=="__main__":
    create_vector_db()
    chain=get_qa_chain()
    print(chain.invoke("What does Joey do for a living?"))