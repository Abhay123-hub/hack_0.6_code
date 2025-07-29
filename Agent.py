from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import requests
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from State import State

from LLMManager import LLMManager

class Agent:
    def __init__(self):
        llm_manager = LLMManager()
        self.llm = llm_manager.get_llm() ## this is the open ai llm
        self.embeddings = llm_manager.get_embeddings()
    def get_relevant_documents(self,state:State):
        url = state.get("documents") ## url of the website from where i can fetch the text data
        with open("file.pdf", "wb") as f:
          f.write(requests.get(url).content)

        loader = PyMuPDFLoader("file.pdf")
        docs = loader.load()
        relevant_documents = []
        # 2. Split the text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name="rag-chrome"
            )
        retriever = vectorstore.as_retriever()
        questions_list = state.get("questions") ## list of user questions
        for question in questions_list:
                document = retriever.invoke(question)
        relevant_documents.append(document)
        
        return {"relevant_documents":relevant_documents}
    
    def get_response(self,state:State):
        prompt = hub.pull("rlm/rag-prompt")
        responses = [] ## empty list where all the responses will be stored
        rag_chain = prompt | self.llm | StrOutputParser()
        questions = state.get("questions") ## list of all questions
        documents = state.get("documents") ## set of all relevant documents
        ## len(questions) == len(documents)
        for i in range(len(questions)):
            question = questions[i] ## any question
            document = documents[i] ## relevant documents of that question
            ## passing both question and documents to rag chain in order to get the response
            response = rag_chain.invoke({"question":question,"context":document})
            responses.append(response)
        return {"responses":responses}

   




