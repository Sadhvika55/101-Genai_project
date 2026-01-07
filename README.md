# 101-Genai_project
# Improved Retrieval-Augmented Generation (RAG) System for LLM Research Papers

## 1. Project Overview
Modern AI assistants use Retrieval-Augmented Generation (RAG) to answer questions using external knowledge sources such as PDFs, webpages, and enterprise documents.  
However, baseline retrieval methods often return loosely relevant or incomplete context, which can lead to hallucinated or unreliable responses.

This project implements and evaluates an **improved retrieval strategy** for a RAG system using a corpus of **top research papers on Large Language Models (LLMs)**.  
The goal is to provide **more coherent, complete, and contextually relevant evidence** to the language model before response generation.

1.Install Dependencies
pip install -r requirements.txt
Required libraries:

sentence-transformers
faiss-cpu
numpy
PyMuPDF
scikit-learn

pip install sentence-transformers ->Installs pre-trained transformer models used to convert text (chunks and queries) into numerical embeddings.
pip install faiss-cpu ->Installs FAISS, which is used to perform fast similarity search on embeddings for retrieval.
pip install numpy ->Provides numerical array operations required to store and process embeddings.
pip install pymupdf ->Enables extraction of text from PDF research papers.
pip install scikit-learn ->Used for utility functions such as similarity calculations and optional evaluation metrics.


2. Prepare the Corpus
Place all LLM research paper PDFs in the papers/ folder.

3. Extract Text from PDFs
python extract_text.py
Extracted text files are stored in the texts/ folder.

4. Chunk the Text
python chunk_text.py
Chunked text is saved in chunks/all_chunks.json.

5. Generate Embeddings
python embeddings.py
Embeddings are stored in chunks/embeddings.npy.

6. Run Baseline Retrieval
python retrieve_chunks.py

7. Run Improved Retrieval
python improved_retrieval.py



## Implementation Details

### Baseline Retrieval
- PDF text is extracted using PyMuPDF.
- Text is split into fixed-size chunks.
- Sentence embeddings are generated using the `all-MiniLM-L6-v2` model.
- FAISS is used to perform similarity search on embeddings.
- Top-K chunks are retrieved based solely on cosine similarity.

### Improved Retrieval Strategy
- Uses the same embedding model and FAISS index for fair comparison.
- Applies semantic filtering to remove weakly relevant chunks.
- Prioritizes definition-level and explanatory content.
- Reduces redundancy by eliminating near-duplicate chunks.
- Returns more coherent and contextually complete chunks.

### Query Processing
- User queries are embedded using the same SentenceTransformer model.
- FAISS retrieves candidate chunks.
- Improved logic re-ranks and filters results before final selection.

### Output
- Retrieved chunks are displayed in ranked order.
- Improved retrieval provides more focused and reliable context for answer generation.
