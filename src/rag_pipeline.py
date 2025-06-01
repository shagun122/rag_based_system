import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
import ast
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROCESSED_DATA_PATH = os.path.join("data", "processed_quotes.csv")
FINE_TUNED_MODEL_PATH = os.path.join("models", "fine_tuned_sentence_transformer")
FAISS_INDEX_PATH = os.path.join("models", "quotes_faiss.index")
QUOTE_DATA_MAP_PATH = os.path.join("models", "quote_data_map.pkl") 

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL_NAME = 'gemini-2.0-flash-001'
else:
    logging.warning("GOOGLE_API_KEY not found in .env file. LLM generation will fail.")
    GEMINI_MODEL_NAME = None


def load_resources():
    """Loads the fine-tuned model, processed data, and FAISS index if it exists."""
    logging.info("Loading resources for RAG pipeline...")

    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        logging.error(f"Fine-tuned model not found at {FINE_TUNED_MODEL_PATH}. Please run fine_tuning.py.")
        raise FileNotFoundError(f"Fine-tuned model not found at {FINE_TUNED_MODEL_PATH}")
    model = SentenceTransformer(FINE_TUNED_MODEL_PATH)
    logging.info("Fine-tuned model loaded.")

    if not os.path.exists(PROCESSED_DATA_PATH):
        logging.error(f"Processed data not found at {PROCESSED_DATA_PATH}. Please run data_preparation.py.")
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}")
    
    quotes_df = pd.read_csv(PROCESSED_DATA_PATH)
    quotes_df['tags'] = quotes_df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # Creating a list of dictionaries for easier mapping later
    quote_data_list = quotes_df.to_dict(orient='records')
    logging.info(f"Processed quotes data loaded ({len(quotes_df)} quotes).")

    # FAISS index
    faiss_index = None
    if os.path.exists(FAISS_INDEX_PATH):
        logging.info(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        logging.info("FAISS index loaded.")
    else:
        logging.info("FAISS index not found. Will build it now.")
        pass 

    return model, quotes_df, quote_data_list, faiss_index

def build_and_save_faiss_index(model: SentenceTransformer, quotes_df: pd.DataFrame, quote_data_list: list, index_path: str, data_map_path: str):
    """Builds FAISS index from quote embeddings and saves it along with data map."""
    logging.info("Building FAISS index...")
    
    passages_to_embed = [
        f"Quote: {row['quote']} Author: {row['author']} Tags: {', '.join(row['tags'] if isinstance(row['tags'], list) else [])}"
        for row in quote_data_list
    ]

    logging.info(f"Generating embeddings for {len(passages_to_embed)} passages...")
    embeddings = model.encode(passages_to_embed, convert_to_tensor=False, show_progress_bar=True) # FAISS needs numpy
    
    if embeddings.ndim == 1: 
        embeddings = np.expand_dims(embeddings, axis=0)

    if embeddings.shape[0] == 0:
        logging.error("No embeddings generated. Cannot build FAISS index.")
        return None

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension) 
    faiss_index.add(embeddings.astype(np.float32)) 
    
    logging.info(f"FAISS index built with {faiss_index.ntotal} vectors.")
    
    # Save the index
    faiss.write_index(faiss_index, index_path)
    logging.info(f"FAISS index saved to {index_path}")

    return faiss_index

def retrieve_quotes(query: str, model: SentenceTransformer, faiss_index: faiss.Index, quote_data_list: list, top_k: int = 5) -> list:
    """Retrieves top_k relevant quotes from the FAISS index."""
    if faiss_index is None:
        logging.error("FAISS index is not available for retrieval.")
        return []
    logging.info(f"Retrieving top {top_k} quotes for query: '{query}'")
    query_embedding = model.encode([query], convert_to_tensor=False) 
    
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k)
    
    retrieved_quotes_details = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        dist = distances[0][i]
        if idx < len(quote_data_list):
            quote_info = quote_data_list[idx]
            retrieved_quotes_details.append({
                "text": quote_info['quote'],
                "author": quote_info['author'],
                "tags": quote_info['tags'],
                "score": 1 / (1 + dist) 
            })
    logging.info(f"Retrieved {len(retrieved_quotes_details)} quotes.")
    return retrieved_quotes_details

def generate_structured_response(query: str, retrieved_quotes: list) -> dict:
    """Generates a structured response using Gemini LLM based on retrieved quotes."""
    if not GEMINI_MODEL_NAME or not GEMINI_API_KEY:
        logging.error("Gemini API key or model name not configured. Cannot generate response.")
        return {"error": "LLM not configured."}
    if not retrieved_quotes:
        logging.warning("No quotes retrieved to generate response.")   

    logging.info("Generating structured response with Gemini LLM...")
    
    context_str = "\n\n".join([
        f"Quote: {q['text']}\nAuthor: {q['author']}\nTags: {', '.join(q['tags'])}\n(Similarity Score: {q['score']:.4f})"
        for q in retrieved_quotes
    ])

    prompt = f"""You are a helpful assistant specializing in quotes. Based on the user's query and the following retrieved quotes, provide a structured JSON response.

User Query: "{query}"

Retrieved Quotes with their similarity scores:
---
{context_str if retrieved_quotes else "No specific quotes were retrieved, but try to answer generally if possible based on the query."}
---

Your JSON response should have the following keys:
- "relevant_quotes": A list of objects, where each object contains the "text", "author", and "tags" of the most relevant quotes from the retrieved set. Include up to 3-5 of the most pertinent ones.
- "authors": A list of unique author names mentioned in your selected "relevant_quotes".
- "tags": A list of unique tags from your selected "relevant_quotes".
- "summary": A concise summary that directly answers or addresses the user's query, synthesized from the provided quotes. If no quotes are directly relevant, state that.

Ensure the output is a single, valid JSON object.
"""

    try:
        llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = llm.generate_content(prompt)
        
        cleaned_response_text = response.text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]
        
        logging.info(f"LLM Raw Response: {cleaned_response_text[:500]}...")
        
     
        json_response = json.loads(cleaned_response_text)
        return json_response
    except Exception as e:
        logging.error(f"Error generating or parsing LLM response: {e}")
        logging.error(f"LLM prompt sent: {prompt[:500]}...")
        logging.error(f"LLM raw text received before error (if any): {response.text if 'response' in locals() else 'N/A'}")
        return {"error": "Failed to get or parse response from LLM.", "details": str(e)}

_model = None
_quotes_df = None
_quote_data_list = None
_faiss_index = None

def initialize_rag_pipeline(force_rebuild_index=False):
    """Initializes all components of the RAG pipeline."""
    global _model, _quotes_df, _quote_data_list, _faiss_index
    
    _model, _quotes_df, _quote_data_list, loaded_index = load_resources()
    
    if force_rebuild_index or not os.path.exists(FAISS_INDEX_PATH):
        logging.info("Building or rebuilding FAISS index as requested or because it's missing.")
        _faiss_index = build_and_save_faiss_index(_model, _quotes_df, _quote_data_list, FAISS_INDEX_PATH, QUOTE_DATA_MAP_PATH)
    else:
        _faiss_index = loaded_index
        if _faiss_index is None: 
             logging.warning("FAISS index file exists but failed to load. Attempting to rebuild.")
             _faiss_index = build_and_save_faiss_index(_model, _quotes_df, _quote_data_list, FAISS_INDEX_PATH, QUOTE_DATA_MAP_PATH)


    if not _faiss_index:
        logging.error("FAISS index could not be loaded or built. RAG pipeline may not function.")
    
    logging.info("RAG pipeline initialized.")


def query_rag(user_query: str, top_k_retrieval: int = 5) -> dict:
    """Handles a user query through the RAG pipeline."""
    global _model, _faiss_index, _quote_data_list

    if not all([_model, _faiss_index, _quote_data_list]):
        logging.error("RAG pipeline not initialized properly. Call initialize_rag_pipeline() first.")
        logging.info("Attempting to initialize RAG pipeline now...")
        initialize_rag_pipeline()
        if not all([_model, _faiss_index, _quote_data_list]):
             return {"error": "RAG pipeline failed to initialize."}


    retrieved_docs = retrieve_quotes(user_query, _model, _faiss_index, _quote_data_list, top_k=top_k_retrieval)
    
    if not retrieved_docs:
        logging.info("No documents retrieved. LLM will be prompted without specific context quotes.")
    
    final_response = generate_structured_response(user_query, retrieved_docs)
    
    return {
        "llm_response": final_response,
        "retrieved_documents": retrieved_docs 
    }


if __name__ == "__main__":
    
    logging.info("Testing RAG pipeline script...")
    if not GEMINI_API_KEY:
        logging.error("GOOGLE_API_KEY is not set. LLM part of the test will fail.")
        logging.error("Please set your GOOGLE_API_KEY in the .env file in the project root.")
        
    initialize_rag_pipeline(force_rebuild_index=False) 

    if _faiss_index: 
        test_queries = [
            "quotes about hope",
            "what did oscar wilde say about life?",
            "courageous women authors",
            "philosophy of time"
        ]

        for t_query in test_queries:
            logging.info(f"\n--- Testing Query: {t_query} ---")
            rag_output = query_rag(t_query)
            import json
            logging.info(f"LLM Response for '{t_query}':\n{json.dumps(rag_output.get('llm_response'), indent=2)}")
            logging.info(f"Retrieved Documents for '{t_query}':")
            for doc in rag_output.get('retrieved_documents', []):
                logging.info(f"  - Score: {doc.get('score', 0):.4f}, Text: {doc.get('text', '')[:70]}...")
            logging.info("-------------------------------\n")
    else:
        logging.error("Could not run tests as FAISS index is unavailable.")
