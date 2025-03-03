import os
import streamlit as st
from tavily import TavilyClient
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langgraph.graph import StateGraph
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

# Disable Streamlit file watcher to avoid torch-related issues
os.environ["STREAMLIT_DISABLE_WATCHDOG"] = "1"

# Set environment variables (replace with your actual API keys in "")
os.environ["TAVILY_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""


class ResearchAgent:
    def __init__(self):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    def search(self, query):
        response = self.client.search(query=query, max_results=5)
        results = [
            Document(
                page_content=result.get("content", ""),
                metadata={
                    "title": result.get("title", "Untitled"),
                    "url": result.get("url", "")
                }
            )
            for result in response.get("results", [])
        ]
        return results


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts)
    
    def embed_query(self, text):
        return self.model.encode([text])[0]


class ProcessingAgent:
    def __init__(self):
        self.embedding_model = SentenceTransformerEmbeddings()
        self.vectorstore = None
    
    def store_data(self, documents):
        if not documents:
            raise ValueError("No documents provided to store.")
        
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings)),
                embedding=self.embedding_model
            )
        else:
            self.vectorstore.add_embeddings(list(zip(texts, embeddings)))
    
    def retrieve_data(self, query):
        if self.vectorstore is None:
            raise ValueError("Vector store has not been initialized with any documents.")
        
        query_embedding = self.embedding_model.embed_query(query)
        return self.vectorstore.similarity_search_by_vector(query_embedding, k=3)


class AnswerDrafterAgent:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('models/gemini-2.0-flash')
    
    def split_text_into_chunks(self, text, max_chunk_size=1000):
        """Split a long text into smaller chunks."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def draft_response(self, documents):
        summaries = []
        
        for doc in documents:
            
            chunks = self.split_text_into_chunks(doc.page_content)
            
            
            for chunk in chunks:
                prompt = f"Provide a concise summary of the following text in bullet points:\n\n{chunk}"
                response = self.model.generate_content(prompt)
                summaries.append(response.text.strip())
        
        
        combined_summary = "\n".join(summaries)
        return combined_summary


@dataclass(frozen=True)
class ResearchState:
    query: str = ""
    results: tuple = ()
    processed_results: tuple = ()
    final_response: str = ""

    def __hash__(self):
        return hash((self.query, self.results, self.processed_results, self.final_response))


graph = StateGraph(ResearchState)

def research_step(state: ResearchState) -> ResearchState:
    agent = ResearchAgent()
    results = agent.search(state.query)
    return ResearchState(query=state.query, results=tuple(results))

def processing_step(state: ResearchState) -> ResearchState:
    agent = ProcessingAgent()
    if not state.results:
        raise ValueError("No results to process.")
    
    agent.store_data(state.results)
    processed_results = agent.retrieve_data(state.query)
    return ResearchState(
        query=state.query,
        results=state.results,
        processed_results=tuple(processed_results)
    )

def answer_step(state: ResearchState) -> ResearchState:
    agent = AnswerDrafterAgent()
    final_response = agent.draft_response(state.processed_results)
    return ResearchState(
        query=state.query,
        results=state.results,
        processed_results=state.processed_results,
        final_response=final_response
    )

graph.add_node("research", research_step)
graph.add_node("processing", processing_step)
graph.add_node("answer", answer_step)

graph.add_edge("research", "processing")
graph.add_edge("processing", "answer")

graph.set_entry_point("research")
graph.set_finish_point("answer")

workflow = graph.compile()

# Step 7: Streamlit UI
st.set_page_config(page_title="Deep Research AI System", layout="wide")
st.title("Deep Research AI System")
st.write("Enter a query to gather and summarize online information.")

query = st.text_input("Enter your query:", "e.g., Latest trends in AI")

if st.button("Search"):
    if query.strip() == "":
        st.error("Please enter a query.")
    else:
        with st.spinner("Processing..."):
            state = ResearchState(query=query)

            try:
                final_state = workflow.invoke(state)
                final_response = final_state.get("final_response", "No response generated.")
                
                
                st.subheader("Generated Summary:")
                st.markdown(final_response.replace("\n", "  \n"))  
            except ValueError as e:
                st.error(f"An error occurred: {str(e)}")
