# Deep Research AI Agentic System – Documentation

**By: Jordan Manoj**

## 1. Introduction
This document provides an overview of the Deep Research AI Agentic System, which utilizes multiple agents to gather and summarize online information. The system is built using Tavily for web crawling, LangGraph for workflow management, and LangChain for handling vector storage and embeddings.

## 2. Libraries Used
The following libraries are used in this project:

- `os` - Manages environment variables.
- `streamlit` - Builds an interactive user interface.
- `tavily` - Provides web crawling capabilities.
- `google-generativeai` - Uses Google Gemini AI for text generation.
- `langchain` - Framework for LLM applications.
- `langgraph` - Manages workflows using state-based execution.
- `sentence-transformers` - Embeddings for semantic search.
- `FAISS` - Efficient similarity search and retrieval.
- `PyMuPDF` - Processes PDF documents.
- `python-docx` - Reads and writes Word documents.
- `dataclasses` - Defines structured data classes.

## 3. System Architecture
The system consists of two primary agents:
- **Research Agent**: Collects data using Tavily.
- **Answer Drafter Agent**: Summarizes the collected data using Google Gemini AI.

## 4. Workflow Using LangGraph & LangChain
The system is structured as a workflow where data flows through different processing stages:
1. **Research Agent** collects relevant online data.
2. **Processing Agent** stores and retrieves relevant data using FAISS.
3. **Answer Drafter Agent** summarizes the processed data using an AI model.

## 5. Functionality of Each Component

### 5.1 Research Agent
The Research Agent uses Tavily's API to search for relevant content online. It extracts page content and metadata (title, URL) from the results.

### 5.2 Processing Agent
The Processing Agent embeds collected documents using Sentence Transformers and stores them in a FAISS vector database. It also retrieves the most relevant results based on query embeddings.

### 5.3 Answer Drafter Agent
This agent utilizes Google Gemini AI to generate a summary of the retrieved documents. It first chunks large documents, then generates concise bullet-point summaries.

## 6. Execution & Expected Output
To run the system, enter a query in the Streamlit interface and click **'Search'**.

**Expected output:**
- A structured summary of the most relevant web pages related to the query.
- Improved accuracy by filtering out irrelevant data before summarization.

## 7. How the Code Works
The system is implemented in Python and follows these steps:

1. **Initialize Environment & API Keys**: Tavily and Gemini API keys are set using environment variables.
2. **Streamlit UI Setup**: The system provides a user-friendly interface for query input.
3. **Research Agent Execution**: Tavily searches for relevant articles and extracts their content.
4. **Processing Agent Execution**: The retrieved data is embedded and stored in a FAISS vector database.
5. **Retrieval & Similarity Search**: The Processing Agent finds the most relevant content for the query.
6. **Answer Drafter Execution**: Gemini AI summarizes the relevant data into a concise response.
7. **Display Results in Streamlit**: The generated summary is formatted and displayed in the UI.

## 8. Approach
The Deep Research AI Agentic System was designed using a modular approach to ensure efficiency, scalability, and accuracy in gathering and summarizing online information. The approach includes:

- **Agent-based Architecture**: Two specialized agents handle research and summarization separately.
- **LangGraph Workflow**: A structured workflow ensures smooth data processing and transition between agents.
- **FAISS Vector Storage**: Enables fast and efficient retrieval of relevant documents.
- **Google Gemini AI for Summarization**: Ensures high-quality, structured, and concise output.

## 9. Solution

### 9.1 Research Agent
This agent uses Tavily's API to search the web and retrieve relevant articles based on the user’s query. The retrieved documents include metadata like title and URL, and their content is stored for processing.

### 9.2 Processing Agent
The Processing Agent converts the documents into embeddings using Sentence Transformers and stores them in a FAISS vector database. This enables efficient retrieval of the most relevant documents based on query similarity.

### 9.3 Answer Drafter Agent
Once the most relevant documents are retrieved, this agent summarizes the content using Google Gemini AI. The system also ensures that long documents are split into manageable chunks before summarization.

### 9.4 LangGraph Workflow
The solution is structured using LangGraph, ensuring a smooth flow from research to processing and finally to summarization. Each stage is executed sequentially with data being refined at each step.

## 10. Conclusion
The Deep Research AI Agentic System is designed to automate and optimize the process of gathering and summarizing online information. By leveraging a multi-agent architecture, FAISS for efficient data retrieval, and Gemini AI for advanced summarization, the system ensures accurate, relevant, and structured information delivery.

The modular design also allows for future improvements, such as adding new agents, improving embeddings, or enhancing the UI for a better user experience. Overall, this approach provides a scalable and efficient solution for automated research and summarization.
