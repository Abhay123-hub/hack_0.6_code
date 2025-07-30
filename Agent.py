from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import requests
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from State import State
import requests
from requests.adapters import HTTPAdapter, Retry
from LLMManager import LLMManager

class Agent:
    def __init__(self):
        llm_manager = LLMManager()
        self.llm = llm_manager.get_llm() ## this is the open ai llm
        self.embeddings = llm_manager.get_embeddings()


    def get_relevant_documents(self, state: dict):
        url = state.get("documents")  # URL of the PDF

        # Set up requests session with retry
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=2)  # Retry 3 times (2s, 4s, 8s)
        session.mount("https://", HTTPAdapter(max_retries=retries))

        try:
            # Download the file with timeout
            response = session.get(url, timeout=60)
            response.raise_for_status()

            # Save PDF locally
            with open("file.pdf", "wb") as f:
                f.write(response.content)

        except requests.exceptions.Timeout:
            raise Exception("⏱️ Downloading the PDF timed out. Please retry.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"⚠️ Error downloading PDF: {e}")

        # Load the PDF
        loader = PyMuPDFLoader("file.pdf")
        docs = loader.load()

        # Split the text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = splitter.split_documents(docs)

        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name="rag-chrome"
        )

        # Retrieve documents
        retriever = vectorstore.as_retriever()
        questions_list = state.get("questions")

        relevant_documents = []
        for question in questions_list:
            document = retriever.invoke(question)
            relevant_documents.append(document)

        return {"relevant_documents": relevant_documents}
    
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

   




