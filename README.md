# Semantic Quote Retrieval System (RAG-Based)

## 🌟 Project Overview

This project implements a sophisticated Retrieval Augmented Generation (RAG) system designed for semantic retrieval of English quotes. Users can input natural language queries (e.g., "find quotes about hope by Maya Angelou"), and the system will return relevant quotes, their authors, associated tags, and a concise AI-generated summary.

The core of the system leverages a sentence embedding model fine-tuned on the quotes dataset for accurate semantic search, combined with a powerful Large Language Model (Gemini) for generating insightful and structured responses.

## ✨ Key Features

*   **Semantic Search:** Understands the meaning behind user queries, not just keywords.
*   **Fine-Tuned Embeddings:** Utilizes a sentence-transformer model specifically fine-tuned on the `Abirate/english_quotes` dataset for enhanced retrieval accuracy.
*   **RAG Pipeline:** Implements a full Retrieval Augmented Generation pipeline for contextual and relevant answer generation.
*   **LLM Integration:** Uses Google's Gemini model to synthesize information and provide structured JSON outputs.
*   **Efficient Indexing:** Employs FAISS for fast and efficient similarity search over quote embeddings.
*   **Quantitative Evaluation:** Includes an evaluation step using the RAGAS framework to assess pipeline performance (metrics like faithfulness and answer relevancy).
*   **Interactive UI:** A user-friendly Streamlit application for easy interaction with the system.

## 🛠️ Technologies Used

*   **Python 3.8+**
*   **Core ML/Data Libraries:**
    *   PyTorch
    *   Transformers (Hugging Face)
    *   Sentence-Transformers (Hugging Face)
    *   Datasets (Hugging Face)
    *   Pandas, NumPy
*   **Vector Store:** FAISS (faiss-cpu)
*   **Large Language Model (LLM):** Google Gemini (via `google-generativeai`)
*   **RAG Evaluation:** RAGAS framework (with Langchain wrappers for LLM integration)
*   **Web Application:** Streamlit
*   **Environment Management:** `venv`
*   **API Key Management:** `python-dotenv`

## 📂 Project Structure

```
quote_rag_system/
├── .env                            # Stores API keys (e.g., GOOGLE_API_KEY) - GITIGNORED
├── .gitignore                      # Specifies intentionally untracked files for Git
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── data/
│   └── processed_quotes.csv        # Cleaned and preprocessed dataset
├── models/
│   ├── fine_tuned_sentence_transformer/ # Saved fine-tuned embedding model
│   ├── quotes_faiss.index          # FAISS index for quote embeddings
│   └── ragas_evaluation_results.csv # RAGAS evaluation scores
│   └── fine_tuned_sentence_transformer_checkpoints/ # Checkpoints from fine-tuning (GITIGNORED)
├── src/
│   ├── __init__.py                 # Makes src a Python package (optional but good practice)
│   ├── data_preparation.py         # Script for data download and preprocessing
│   ├── fine_tuning.py              # Script for fine-tuning the embedding model
│   ├── rag_pipeline.py             # Core RAG logic (retrieval and generation)
│   ├── evaluation.py               # Script for RAGAS evaluation
│   └── app.py                      # Streamlit application
└── venv/                           # Python virtual environment (GITIGNORED)
```

## 🚀 Setup and Installation

1.  **Clone the Repository (if applicable, once it's on GitHub):**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create and Activate a Python Virtual Environment:**
    *   It's highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    ```
    *   Activate it:
        *   Windows: `venv\Scripts\activate`
        *   macOS/Linux: `source venv/bin/activate`

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up API Key:**
    *   Create a file named `.env` in the project root directory (`quote_rag_system/`).
    *   Add your Google API key to this file:
        ```
        GOOGLE_API_KEY="your_google_api_key_here"
        ```
    *   *(Optional: If RAGAS evaluation still defaults to OpenAI and you wish to use it for RAGAS judgments, you might also need to set `OPENAI_API_KEY="your_openai_key_here"` in the `.env` file or as an environment variable.)*

## ⚙️ Running the System Components

The project is divided into several runnable scripts. It's generally recommended to run them in sequence, especially for the first time.

1.  **Data Preparation:**
    *   This script downloads the dataset, preprocesses it, and saves it to `data/processed_quotes.csv`.
    *   Run from the project root (`quote_rag_system/`):
        ```bash
        python src/data_preparation.py
        ```

2.  **Model Fine-Tuning:**
    *   This script loads the processed data and fine-tunes the sentence embedding model. The fine-tuned model is saved to `models/fine_tuned_sentence_transformer/`.
    *   This step can take several minutes.
    *   Run from the project root:
        ```bash
        python src/fine_tuning.py
        ```

3.  **RAG Pipeline Initialization & FAISS Index Building:**
    *   The `rag_pipeline.py` script, when run directly, will initialize the pipeline. This includes loading the fine-tuned model and building the FAISS index (`models/quotes_faiss.index`) if it doesn't already exist. The index building process (embedding all quotes) can also take a few minutes on the first run.
    *   It also runs some test queries.
    *   Run from the project root:
        ```bash
        python src/rag_pipeline.py
        ```

4.  **RAG Evaluation (Optional but Recommended):**
    *   This script evaluates the RAG pipeline using RAGAS and saves the results to `models/ragas_evaluation_results.csv`.
    *   This step involves LLM calls and can take time.
    *   Run from the project root:
        ```bash
        python src/evaluation.py
        ```

5.  **Running the Streamlit Web Application:**
    *   This is the main interface for interacting with the system.
    *   Ensure all previous steps (especially data prep, fine-tuning, and initial RAG pipeline run for index creation) have been completed.
    *   Run from the project root:
        ```bash
        streamlit run src/app.py
        ```
    *   If you encounter a `RuntimeError` related to `torch.classes` and Streamlit's file watcher, try running with the watcher disabled:
        *   Command Prompt: `set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none && streamlit run src/app.py`
        *   PowerShell: `$env:STREAMLIT_SERVER_FILE_WATCHER_TYPE="none"; streamlit run src/app.py`

## 🧠 RAG Pipeline Explained

The core `rag_pipeline.py` script orchestrates the following:

1.  **Initialization (`initialize_rag_pipeline`):**
    *   Loads the fine-tuned sentence embedding model (`SentenceTransformer`).
    *   Loads the processed quotes data.
    *   Loads an existing FAISS index or builds a new one by:
        *   Encoding all quotes into vector embeddings using the fine-tuned model.
        *   Storing these embeddings in a FAISS index for efficient similarity search.
    *   These components are stored in module-level global variables to be reused by the Streamlit app without reloading.

2.  **Query Processing (`query_rag`):**
    *   **Retrieval (`retrieve_quotes`):**
        *   The user's natural language query is encoded into an embedding using the same fine-tuned model.
        *   The FAISS index is searched to find the `top_k` quotes whose embeddings are most similar (closest in vector space) to the query embedding.
        *   Similarity scores are calculated.
    *   **Generation (`generate_structured_response`):**
        *   The original query and the text of the retrieved quotes (along with their scores) are formatted into a detailed prompt.
        *   This prompt is sent to the Gemini LLM.
        *   The LLM is instructed to provide a JSON response containing a summary, a curated list of relevant quotes, authors, and tags.
    *   The `query_rag` function returns both the LLM's structured response and the list of initially retrieved documents.

## 📊 Dataset

*   **Source:** `Abirate/english_quotes` from Hugging Face Datasets.
*   **Content:** A collection of English quotes with associated authors and tags.
*   **Preprocessing:** Includes lowercasing, whitespace stripping, and handling of missing values for authors and tags.

## 💡 Model Fine-Tuning

*   **Base Model:** `all-MiniLM-L6-v2` (a fast and effective sentence-transformer).
*   **Task:** To make the model better at understanding the semantic relationships between queries (composed of authors, tags, or topics) and the detailed quote passages.
*   **Method:**
    *   Synthetic queries are generated (e.g., "Author quotes about Tag").
    *   `InputExample` pairs of `(query, detailed_quote_passage)` are created.
    *   `MultipleNegativesRankingLoss` is used to train the model to map related queries and passages closer together in the embedding space.

## 📈 Evaluation

*   **Framework:** RAGAS (Retrieval Augmented Generation Assessment).
*   **Metrics Used (example):**
    *   `faithfulness`: Measures if the generated answer is factually consistent with the retrieved context.
    *   `answer_relevancy`: Measures how relevant the generated answer is to the user's query.
    *   *(Note: `context_precision` and `context_recall` were initially planned but required ground truth data not generated in this project, so they were excluded from the final RAGAS run.)*
*   **Process:** A set of evaluation questions are run through the RAG pipeline, and the generated answers and retrieved contexts are fed to RAGAS for scoring.
*   **LLM for Judging:** RAGAS metrics like faithfulness and answer relevancy use an LLM for judgment. This project configures RAGAS to use the Gemini model via a Langchain wrapper for these judgments.

## 🔮 Potential Future Improvements

*   Implement more sophisticated query generation for fine-tuning.
*   Create a ground truth dataset to enable more RAGAS metrics (like `context_precision`, `context_recall` against ground truth documents, `answer_correctness`).
*   Experiment with different base models for fine-tuning or different LLMs for generation.
*   Optimize the FAISS index (e.g., using `IndexIVFFlat`) for even larger datasets.
*   Add user authentication and history to the Streamlit app.
*   Deploy the Streamlit application to a cloud platform.

---

*This project was developed as part of an assignment.*
