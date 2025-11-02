## This assessment will validate the learner's ability apply LlamaIndex

- Loading CV Files: This involves reading CV files from [dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- You can choose up to 20-30 CVs.
- Split CV into small meaningful chunks.
- Generating Embeddings: Convert the parsed data into numerical representations
(embeddings) that can be easily processed by machine learning algorithms. This typically
involves using techniques like word embeddings or sentence embeddings.
- Storing Embeddings in a Vector Database: Save the generated embeddings into a vector
database. As a vector store, you can choose PostgreSQL, ChromaDB, FAISS, etc,
- Retrieving Candidate Details: Extract and display specific information about each
candidate, such as name, profession, and years of commercial experience.
- Generating Experience Summary: Based on the parsed data and embeddings, generate a
summary of each candidate’s strongest skills and professional highlights.

### Important note:
- The task should be done using LlamaIndex.

### Expected outcome:
- The repository contains a straightforward web application that lists candidates. Users can click on
any candidate to view detailed information and a summary of their profile.


### Code

```bash
pip install llama-index
pip install llama-index-node-parser-slide
pip install llama-index-vector-stores-chroma
pip install llama-index-embeddings-huggingface
pip install ipython
pip install "fastapi[standard]"
```

