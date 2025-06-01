import streamlit as st
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from rag_pipeline import initialize_rag_pipeline, query_rag
except ImportError:
    logging.error("Failed to import from rag_pipeline. Ensure rag_pipeline.py is in the same directory or PYTHONPATH is set correctly.")
    st.error("Critical error: Could not load the RAG pipeline module. Please check the logs.")
    st.stop()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(
    page_title="Semantic Quote Finder",
    page_icon="📚",
    layout="wide"
)

#Loading RAG Pipeline
@st.cache_resource
def load_pipeline_cached(): 
    """Loads and initializes the RAG pipeline components."""
    logging.info("Attempting to initialize RAG pipeline for Streamlit app...")
    try:
        initialize_rag_pipeline(force_rebuild_index=False)
        logging.info("RAG pipeline initialized successfully for Streamlit app.")
        return True
    except Exception as e:
        logging.error(f"Error initializing RAG pipeline in Streamlit: {e}", exc_info=True)
        st.error(f"Failed to initialize the RAG pipeline: {e}")
        return False

# Main Application UI
st.title("📚 Semantic Quote Finder")
st.markdown("Enter a query to find relevant quotes, their authors, tags, and a summary.")

pipeline_loaded = load_pipeline_cached()

if pipeline_loaded:
    user_query = st.text_input(
        "Enter your query (e.g., 'quotes about hope by inspiring women'):", 
        "", 
        key="query_input_main"
    )

    if st.button("Search Quotes", key="search_button_main"):
        if user_query:
            with st.spinner("Searching for quotes and generating insights..."):
                try:
                    rag_output = query_rag(user_query)
                    llm_response = rag_output.get("llm_response")
                    retrieved_documents = rag_output.get("retrieved_documents")

                    if llm_response and "error" not in llm_response:
                        st.subheader("🔍 Summary & Insights")
                        st.markdown(llm_response.get("summary", "No summary provided."))

                        st.subheader("✨ Relevant Quotes (Selected by LLM)")
                        relevant_llm_quotes = llm_response.get("relevant_quotes", [])
                        if relevant_llm_quotes:
                            for quote_info in relevant_llm_quotes:
                                with st.container(border=True):
                                    st.markdown(f"**\"{quote_info.get('text', 'N/A')}\"**")
                                    st.caption(f"— {quote_info.get('author', 'Unknown Author')} | Tags: {', '.join(quote_info.get('tags', []))}")
                        else:
                            st.info("The LLM did not select any specific quotes for the summary based on the retrieved context.")

                        authors = llm_response.get("authors", [])
                        tags = llm_response.get("tags", [])
                        if authors:
                            st.markdown(f"**Authors Mentioned:** {', '.join(authors)}")
                        if tags:
                            st.markdown(f"**Associated Tags:** {', '.join(tags)}")

                        st.subheader("📚 Top Retrieved Source Quotes (Before LLM Processing)")
                        if retrieved_documents:
                            for doc in retrieved_documents:
                                with st.expander(f"Quote by {doc.get('author', 'Unknown')} (Score: {doc.get('score', 0):.4f})"):
                                    st.markdown(f"**\"{doc.get('text', 'N/A')}\"**")
                                    st.caption(f"Tags: {', '.join(doc.get('tags', []))}")
                        else:
                            st.info("No documents were initially retrieved by the semantic search for this query.")

                    elif llm_response and "error" in llm_response:
                        st.error(f"Error from RAG pipeline: {llm_response.get('error')}")
                        if llm_response.get('details'):
                            st.error(f"Details: {llm_response.get('details')}")
                    else:
                        st.error("An unexpected error occurred, or no response was generated.")
                
                except Exception as e:
                    logging.error(f"Error during Streamlit query processing: {e}", exc_info=True)
                    st.error(f"An application error occurred: {e}")
        else:
            st.warning("Please enter a query to search.")
else:
    st.error("The RAG pipeline could not be loaded. The application cannot function. Please check the server logs for more details.")

st.sidebar.header("About")
st.sidebar.info(
    "This application uses a Retrieval Augmented Generation (RAG) pipeline "
    "to find quotes based on semantic meaning. It features a fine-tuned "
    "sentence embedding model and a Large Language Model (Gemini) for generation."
)
st.sidebar.markdown("---")
