from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

SUGGEST_CHAIN = None


def initialize_suggest_chain():
    global SUGGEST_CHAIN
    if SUGGEST_CHAIN is not None:
        return SUGGEST_CHAIN
    
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        loader1 = PyPDFLoader(os.path.join(base_path, "books/plant_disease_knowledge_base.pdf"))
        loader2 = PyPDFLoader(os.path.join(base_path, "books/Plant-Pathology-and-Disease-Management.pdf"))
        loader3 = PyPDFLoader(os.path.join(base_path, "books/westcotts-plant-disease-handbook-9781402045844-9781402045851_compress.pdf"))

        all_docs = loader1.load() + loader2.load() + loader3.load()

        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        splitted_docs = SemanticChunker(embedding, breakpoint_threshold_type="percentile").split_documents(all_docs)

        vector_store = FAISS.from_documents(splitted_docs, embedding)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        prompt_template = """You are a plant disease expert helping Asian farmers understand crop diseases.

Rules:
- Use simple language
- Use short sentences
- Focus on practical actions
- Use bullet points
- Explain like talking to a village farmer

Disease: {disease}

Using the following information:
{retrieved_docs}

Please explain the {context} of {disease} in simple language a farmer can understand.
Keep your answer clear, short, and actionable."""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["disease", "context", "retrieved_docs"]
        )

        llm = ChatGroq(model_name="llama-3.3-70b-versatile")

        SUGGEST_CHAIN = (
            {
                "retrieved_docs": RunnableLambda(lambda x: x["disease"] + " " + x["context"]) | retriever | RunnableLambda(format_docs),
                "disease": RunnableLambda(lambda x: x["disease"]),
                "context": RunnableLambda(lambda x: x["context"]),
            }
            | prompt
            | llm
        )
        return SUGGEST_CHAIN
    except Exception as e:
        print(f"Error initializing suggest chain: {e}")
        return None


def get_disease_details(disease: str, context: str) -> dict:
    chain = initialize_suggest_chain()
    if chain is None:
        return {"error": "Suggest chain not initialized"}
    
    try:
        output = chain.invoke({"disease": disease, "context": context})
        return {
            "disease": disease,
            "context": context,
            "details": output.content
        }
    except Exception as e:
        return {"error": str(e)}
