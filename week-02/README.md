# OpportunityMatch AI — Week 02

## Project overview
OpportunityMatch AI is a prototype semantic search engine for student opportunities. It helps users search a small dataset of hackathons, internships, workshops, scholarships, fellowships, bootcamps, and projects using natural-language queries.

## Problem
Students may not know the exact title or keywords of an opportunity. A keyword-only search can miss relevant results when the query uses different wording.

## Solution
The project prepares opportunity text, converts it into embedding vectors, stores vectors and metadata in ChromaDB, and retrieves relevant records for a query. A hybrid score combines semantic similarity and simple keyword overlap.

## Features
- CSV loading and schema validation
- Text normalization and basic HTML cleanup
- Text chunking with `RecursiveCharacterTextSplitter`
- Embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Persistent local vector storage with ChromaDB
- Top-k semantic retrieval
- Hybrid semantic and lexical ranking
- Optional category filtering
- Initial ranking evaluation using Recall@3 and Mean Reciprocal Rank (MRR)

## Project structure

```text
week-02/
├── data/
│   └── opportunities.csv
├── task.py
├── task.ipynb
├── README.md
├── requirements.txt
└── chroma_db/       
```

## Requirements
- Python 3.11 or 3.12 is a practical choice if a dependency does not provide a compatible wheel for your Python version.
- Internet access for the first download of the embedding model.
- VS Code with the Python and Jupyter extensions if you want to run the notebook in VS Code.

Install the packages in your active virtual environment:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install "pandas>=2.2.3" langchain-text-splitters chromadb sentence-transformers
```

After installation succeeds, create a reproducible dependency list:

```bash
python -m pip freeze > requirements.txt
```

## Run the application
Open a terminal in the `week-02` directory and run:

```bash
python task.py
```

The first run may take time because the Sentence Transformers model must be downloaded. The script then prepares the data, builds the local ChromaDB collection, demonstrates several queries, and runs the initial evaluation.

## Run the notebook
1. Open `task.ipynb` in VS Code.
2. Select the Python interpreter associated with the project's `.venv`.
3. Run notebook cells from top to bottom.
4. Run the search and evaluation sections only after `python task.py` has completed the first model download and created the ChromaDB collection.

Jupyter notebooks combine executable code with explanatory Markdown, making them useful for documenting experiments and findings. See the [VS Code Jupyter documentation](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

## Evaluation
The notebook reports Recall@3 and MRR using labeled example queries. The relevance labels are initial estimates and must be checked manually against the actual results before reporting metrics.

## Limitations
- The dataset is a small synthetic example, not a live list of real opportunities.
- Results do not verify that an opportunity is still open, legitimate, or suitable for a particular applicant.
- Keyword overlap is a simple baseline; it is not a trained reranker.
- Evaluation on a small number of queries is only an initial signal, not proof of general performance.

## Future work
- Add verified source URLs, deadlines, locations, and last-checked timestamps.
- Increase the dataset size and improve manually labeled test queries.
- Compare semantic-only and hybrid ranking on the same queries.
- Add a cross-encoder reranker and measure whether it improves Recall@3 and MRR.
- Add automated tests and, later, an API or user interface.

## Author
Tesnim Abdi
