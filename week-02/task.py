
from pathlib import Path
import html
import logging
import re
import unicodedata

import chromadb
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# -------------------- CONFIGURATION --------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "opportunities.csv"
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "student_opportunities"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 60

REQUIRED_COLUMNS = {
    "id", "title", "category",
    "description", "skills", "eligibility"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------- DATA LOADING --------------------

def clean_text(value):
    """Normalize text and remove HTML markup."""
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(path=DATA_PATH):
    """Load and validate the opportunities CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("IDs must be present and unique.")

    for column in REQUIRED_COLUMNS - {"id"}:
        df[column] = df[column].apply(clean_text)

    df["id"] = df["id"].astype(str)

    df = df[
        (df["title"] != "") &
        (df["description"] != "")
    ].copy()

    if df.empty:
        raise ValueError("No valid opportunities remain.")

    return df


# -------------------- TEXT CHUNKING --------------------

class TextProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " "]
        )

    def process(self, row):
        text = (
            f"Title: {row['title']}\n"
            f"Category: {row['category']}\n"
            f"Description: {row['description']}\n"
            f"Skills: {row['skills']}\n"
            f"Eligibility: {row['eligibility']}"
        )

        chunks = self.splitter.split_text(text)
        return chunks or [text]


# -------------------- SEARCH ENGINE --------------------

class OpportunitySearchEngine:
    def __init__(self):
        logger.info("Loading embedding model: %s", MODEL_NAME)
        self.model = SentenceTransformer(MODEL_NAME)

        self.client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )

        self.collection = None
        self.processor = TextProcessor()

    def ingest_data(self):
        """Create embeddings and save opportunity chunks."""
        df = load_data(DATA_PATH)

        documents = []
        ids = []
        metadatas = []

        for _, row in df.iterrows():
            chunks = self.processor.process(row)

            for chunk_number, chunk in enumerate(chunks):
                documents.append(chunk)
                ids.append(
                    f"opportunity-{row['id']}-chunk-{chunk_number}"
                )

                metadatas.append({
                    "opportunity_id": row["id"],
                    "title": row["title"],
                    "category": row["category"],
                    "skills": row["skills"],
                    "eligibility": row["eligibility"],
                })

        if not documents:
            raise ValueError("No text chunks were generated.")

        logger.info("Generating %s embeddings...", len(documents))
        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True
        ).tolist()

        # Rebuild the collection so updated CSV data is reflected.
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        batch_size = 100

        for start in range(0, len(documents), batch_size):
            end = start + batch_size

            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end]
            )

        stats = {
            "opportunities": len(df),
            "chunks": len(documents),
            "categories": df["category"].nunique()
        }

        logger.info("Database ingestion complete: %s", stats)
        return stats

    @staticmethod
    def lexical_score(query, document):
        """Calculate simple keyword overlap."""
        query_words = set(re.findall(r"\w+", query.lower()))
        document_words = set(re.findall(r"\w+", document.lower()))

        if not query_words:
            return 0.0

        return len(query_words & document_words) / len(query_words)

    def search(self, query, top_k=TOP_K, category=None):
        """Rank opportunities using semantic and keyword relevance."""
        query = clean_text(query)

        if not query:
            raise ValueError("Search query cannot be empty.")

        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")

        if self.collection is None:
            self.collection = self.client.get_collection(
                COLLECTION_NAME
            )

        where = {"category": category} if category else None

        if where:
            filtered = self.collection.get(
                where=where,
                include=[]
            )
            available = len(filtered["ids"])
        else:
            available = self.collection.count()

        if available == 0:
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        response = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(10, available),
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        ranked = {}

        for document, metadata, distance in zip(
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0]
        ):
            semantic = max(0.0, min(1.0, 1.0 - distance))
            lexical = self.lexical_score(query, document)

            # Hybrid ranking: semantic similarity + keyword overlap.
            score = 0.8 * semantic + 0.2 * lexical
            opportunity_id = metadata["opportunity_id"]

            result = {
                **metadata,
                "matched_text": document,
                "semantic_score": semantic,
                "lexical_score": lexical,
                "hybrid_score": score
            }

            if (
                opportunity_id not in ranked
                or score > ranked[opportunity_id]["hybrid_score"]
            ):
                ranked[opportunity_id] = result

        results = sorted(
            ranked.values(),
            key=lambda item: item["hybrid_score"],
            reverse=True
        )

        return results[:top_k]

    @staticmethod
    def display_results(results):
        if not results:
            print("No matching opportunities found.")
            return

        for rank, item in enumerate(results, start=1):
            print(f"\n{rank}. {item['title']}")
            print(f"   Category: {item['category']}")
            print(f"   Skills: {item['skills']}")
            print(f"   Eligibility: {item['eligibility']}")
            print(f"   Semantic score: {item['semantic_score']:.3f}")
            print(f"   Hybrid score: {item['hybrid_score']:.3f}")
            print(f"   Matched text: {item['matched_text']}")


# -------------------- EVALUATION --------------------

def evaluate(engine):
    """Measure ranking quality using manually labeled examples."""
    test_cases = [
        {
            "query": "I want to learn AI with Python",
            "relevant": {"1", "2", "15", "18"}
        },
        {
            "query": "I want to learn ethical hacking",
            "relevant": {"3"}
        },
        {
            "query": "I want to build websites with React",
            "relevant": {"4"}
        },
        {
            "query": "I need funding for student research",
            "relevant": {"12", "17"}
        },
        {
            "query": "I want mentorship for women in technology",
            "relevant": {"6"}
        },
        {
            "query": "I want to learn SQL and database design",
            "relevant": {"19", "10", "5"}
        },
        {
            "query": "I want to contribute code to public projects",
            "relevant": {"11"}
        },
        {
            "query": "I want an internship in software development",
            "relevant": {"16", "5", "10", "15"}
        },
        {
            "query": "I want to build mobile applications",
            "relevant": {"9", "1"}
        },
        {
            "query": "I want to develop a technology startup",
            "relevant": {"20", "1"}
        }
    ]

    recalls = []
    reciprocal_ranks = []

    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50)

    for case in test_cases:
        results = engine.search(case["query"], top_k=3)
        result_ids = [item["opportunity_id"] for item in results]
        relevant = case["relevant"]

        hits = len(set(result_ids) & relevant)
        recall = hits / len(relevant)

        reciprocal_rank = 0.0
        for rank, result_id in enumerate(result_ids, start=1):
            if result_id in relevant:
                reciprocal_rank = 1 / rank
                break

        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)

        print(f"\nQuery: {case['query']}")
        print(f"Returned IDs: {result_ids}")
        print(f"Recall@3: {recall:.3f}")
        print(f"Reciprocal rank: {reciprocal_rank:.3f}")

    print(f"\nMean Recall@3: {sum(recalls) / len(recalls):.3f}")
    print(
        "Mean Reciprocal Rank: "
        f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}"
    )


# -------------------- MAIN PROGRAM --------------------

def main():
    print("=" * 50)
    print("OPPORTUNITYMATCH AI | WEEK 02")
    print("=" * 50)

    df = load_data()

    print(f"Dataset: {DATA_PATH.name}")
    print(f"Total opportunities: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print("\nDataset preview:")
    print(df.head(5).to_string(index=False))

    engine = OpportunitySearchEngine()

    print("\nBuilding the searchable database...")
    stats = engine.ingest_data()
    print("Ingestion statistics:", stats)

    queries = [
        "I want to learn AI with Python",
        "Find cybersecurity opportunities for students",
        "I need funding for student research",
        "I want to build websites using React"
    ]

    for query in queries:
        print("\n" + "=" * 50)
        print("Search query:", query)
        results = engine.search(query, top_k=3)
        engine.display_results(results)

    evaluate(engine)


if __name__ == "__main__":
    main()
