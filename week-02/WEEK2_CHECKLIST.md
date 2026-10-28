# Week 02 — Work Checklist

## While the embedding model downloads
- [ ] Leave the running terminal alone while the download is progressing.
- [ ] Create `task.ipynb` from the supplied notebook.
- [ ] Add the supplied `README.md`.
- [ ] Add `.gitignore` to the repository root (merge with any existing rules rather than overwriting useful rules).
- [ ] Confirm `opportunities.csv` has 20 rows and six columns.
- [ ] Do not start another model download from the notebook while `task.py` is still downloading.

## After the current `python task.py` run finishes
- [ ] Confirm ingestion statistics show 20 opportunities.
- [ ] Confirm the script prints search results and evaluation metrics.
- [ ] If it errors, save the full traceback before changing code.
- [ ] In VS Code, open `task.ipynb` and select the same `.venv` Python interpreter.
- [ ] Run notebook cells from top to bottom after the model and database are ready.
- [ ] Review whether each search result is genuinely relevant.
- [ ] Verify/correct the labeled relevant IDs before trusting evaluation metrics.
- [ ] Create `requirements.txt` only after package installation succeeds: `python -m pip freeze > requirements.txt`.
- [ ] Run `git status` and confirm `.venv/` and `chroma_db/` are ignored.
- [ ] Commit the Week 02 files to GitHub after the code runs successfully.

## Later improvements
- [ ] Compare semantic-only ranking with hybrid ranking.
- [ ] Add a cross-encoder reranker.
- [ ] Add more labeled queries and compare Recall@3 and MRR.
- [ ] Add tests for empty queries, duplicate IDs, missing columns, and unknown categories.
