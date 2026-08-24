#!/usr/bin/env python3
"""Build the ai_tutor_knowledge vector store once, so notebooks load it instead
of re-embedding. One-shot tool: delete it when the stores are built.

Ingestion is the updated notebooks' own method, unchanged (10-Adding_Reranking,
Larger_Context_Larger_N, Metadata_Filtering): chunk() at 512 tokens with a
128-token overlap (cl100k_base), chunk ids "{doc_id}-{i}", metadata
title/url/source, embed() with Gemini task_type RETRIEVAL_DOCUMENT +
output_dimensionality (or the plain OpenAI call), ChromaDB PersistentClient
with a cosine collection named "ai_tutor_knowledge", upsert in batches of 50.
Keys come from .env. 

Recipe note (Aug 2026): three build recipes were measured on this corpus with
the course's hit-rate/MRR eval — this course-default one, a heading-aware
800/100 variant, and a heading-aware 512/100 + title-prefix variant. The
course default measured best, so it is what this script builds. The remaining
headroom on identifier-style questions is retrieval-time (BM25 + RRF hybrid,
reranking — course Section 7), not chunking.

The three configurations, run from the repo root:

  python scripts/build_vector_store.py --model gemini-embedding-001 --dimensions 3072
  python scripts/build_vector_store.py --model gemini-embedding-001 --dimensions 1536 --normalize
  python scripts/build_vector_store.py --model text-embedding-3-small

Same dataset, same chunks, same ids every run — only the embedding differs.
Each run writes notebooks/prebuilt_stores/<config>/{chroma/, manifest.json}
plus a <config>-<dataset-hash>.zip, which Prebuilt_Store_Bakeoff.ipynb loads.
A rate-limited or interrupted run is safe: rerun the same command and batches
already in the store are skipped, not re-embedded. A store left over from a
DIFFERENT chunking recipe is detected (manifest recipe, then stored-text
comparison) and refused — delete its folder first; stale and fresh chunks
never mix.

Normalisation (verified against provider docs, Aug 2026): gemini-embedding-001's
3072-wide default arrives unit-normalised, but smaller widths are truncations
that must be L2-normalised by hand — hence --normalize for the 1536 run
(https://ai.google.dev/gemini-api/docs/embeddings). OpenAI embeddings arrive
"normalized to length 1", so the OpenAI run needs nothing extra
(https://developers.openai.com/api/docs/guides/embeddings).
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import chromadb
import numpy as np
import tiktoken
from dotenv import load_dotenv

load_dotenv()

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DIMS = {"gemini-embedding-001": 3072, "text-embedding-3-small": 1536}
PRICE_PER_MTOK = {"gemini-embedding-001": 0.15, "text-embedding-3-small": 0.02}  # USD, Aug 2026
CHUNK_SIZE, CHUNK_OVERLAP = 512, 128   # the course default — best of the three measured recipes
RECIPE = {"mode": "fixed_token_window", "chunk_size": CHUNK_SIZE,
          "chunk_overlap": CHUNK_OVERLAP, "tokenizer": "cl100k_base"}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--model", required=True, choices=sorted(DEFAULT_DIMS))
parser.add_argument("--dimensions", type=int, help="output width (default: the model's native width)")
parser.add_argument("--normalize", action="store_true", help="L2-normalise vectors (required for Gemini below 3072)")
parser.add_argument("--input", type=Path,
                    default=REPO.parent / "tai-dataset" / "Full_Stack_AI_Engineering" / "ai_tutor_knowledge.jsonl")
parser.add_argument("--output", type=Path, default=REPO / "notebooks" / "prebuilt_stores")
parser.add_argument("--limit", type=int, help="only the first N documents (smoke test)")
parser.add_argument("--dry-run", action="store_true", help="count chunks/tokens/cost only; no API calls")
args = parser.parse_args()

dims = args.dimensions or DEFAULT_DIMS[args.model]
gemini = args.model == "gemini-embedding-001"
if gemini and dims != 3072 and not args.normalize:
    raise SystemExit(f"Gemini at {dims} dims is a truncation and NOT unit length — rerun with --normalize.")
if args.normalize and (not gemini or dims == 3072):
    raise SystemExit("This configuration already arrives unit-normalised from the API — drop --normalize.")

# --- chunk(), exactly as in the updated notebooks ------------------------------
ENC = tiktoken.get_encoding("cl100k_base")


def chunk(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into token-based chunks; consecutive chunks share chunk_overlap tokens."""
    tokens = ENC.encode(text)
    chunks = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        chunks.append(ENC.decode(window))
        if start + chunk_size >= len(tokens):
            break  # the final window reached the end of the text
    return chunks


# --- load + chunk the corpus (course ids and metadata) -------------------------
docs = [json.loads(line) for line in open(args.input, encoding="utf-8")][: args.limit]
records = []
for doc in docs:
    for i, piece in enumerate(chunk(doc["content"])):
        records.append({"id": f"{doc['doc_id']}-{i}", "text": piece,
                        "title": doc["name"], "url": doc["url"], "source": doc["source"]})

total_tokens = sum(len(ENC.encode(r["text"].replace("\n", " "))) for r in records)
est_cost = total_tokens / 1e6 * PRICE_PER_MTOK[args.model]
slug = f"{args.input.stem}-{args.model}-{dims}d" + ("-norm" if args.normalize else "") \
       + (f"-limit{args.limit}" if args.limit else "")
print(f"{len(docs)} documents → {len(records)} chunks ({CHUNK_SIZE}-token windows, {CHUNK_OVERLAP} overlap) "
      f"| ~{total_tokens:,} tokens | est. ${est_cost:.2f} | {args.model} @ {dims}d"
      f"{' + L2 normalisation' if args.normalize else ''} → {args.output / slug}")
if args.dry_run:
    print("--dry-run: no API calls made, nothing written.")
    raise SystemExit(0)

# --- embed(), exactly as in the notebooks (document side only) -----------------
if gemini:
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client()

    def embed_documents(texts):
        result = client.models.embed_content(
            model=args.model,
            contents=[t.replace("\n", " ") for t in texts],
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT",
                                                  output_dimensionality=dims),
        )
        return [e.values for e in result.embeddings]
else:
    from openai import OpenAI

    client = OpenAI()

    def embed_documents(texts):
        result = client.embeddings.create(model=args.model,
                                          input=[t.replace("\n", " ") for t in texts])
        return [d.embedding for d in sorted(result.data, key=lambda d: d.index)]


# --- ingest: chunk → embed → upsert, batches of 50 (course convention) ---------
store_dir = args.output / slug
collection = chromadb.PersistentClient(path=str(store_dir / "chroma")).get_or_create_collection(
    name="ai_tutor_knowledge",
    metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine similarity
)

# Refuse to mix chunking recipes. Primary check: the recipe a completed build
# recorded in its manifest. Fallback (an interrupted build has no manifest):
# compare stored chunk TEXT against this run's chunk for the same id — id
# names overlap across recipes, so only content tells the truth.
if collection.count():
    old_manifest = store_dir / "manifest.json"
    if old_manifest.exists():
        old_recipe = json.loads(old_manifest.read_text()).get("chunking", {})
        stale = {k: old_recipe.get(k) for k in RECIPE} != RECIPE
    else:
        expected = {r["id"]: r["text"] for r in records}
        got = collection.get(limit=50, include=["documents"])
        stale = any(expected.get(cid) != txt for cid, txt in zip(got["ids"], got["documents"]))
    if stale:
        raise SystemExit(
            f"{store_dir} holds chunks from a DIFFERENT chunking recipe.\n"
            f"Delete the old store and zip, then rerun this same command:\n"
            f"  rm -rf '{store_dir}' && rm -f '{args.output}/{slug}'-*.zip"
        )

t0 = time.time()
for start in range(0, len(records), 50):
    batch = records[start : start + 50]
    ids = [r["id"] for r in batch]
    if len(collection.get(ids=ids)["ids"]) == len(ids):
        continue  # already embedded by an earlier (interrupted) run — costs nothing
    for attempt in range(6):  # survive rate limits; a hard failure still resumes on rerun
        try:
            vectors = embed_documents([r["text"] for r in batch])
            break
        except Exception as exc:
            if attempt == 5:
                raise
            print(f"  retry {attempt + 1}/5 after {type(exc).__name__}: {exc}")
            time.sleep(2 ** attempt)
    assert len(vectors) == len(batch) and all(len(v) == dims for v in vectors)
    if args.normalize:
        arr = np.asarray(vectors, dtype=np.float64)
        vectors = (arr / np.linalg.norm(arr, axis=1, keepdims=True)).tolist()
    collection.upsert(
        ids=ids,
        documents=[r["text"] for r in batch],
        embeddings=vectors,
        metadatas=[{"title": r["title"], "url": r["url"], "source": r["source"]} for r in batch],
    )
    if (start // 50) % 10 == 0:
        print(f"  [{min(start + 50, len(records))}/{len(records)} chunks] {time.time() - t0:.0f}s")

# --- verify, record, zip -------------------------------------------------------
count = collection.count()
assert count == len(records), f"store holds {count} vectors, expected {len(records)} — rerun to resume"
sample = collection.get(limit=100, include=["embeddings"])["embeddings"]
norms = np.linalg.norm(np.asarray(sample, dtype=np.float64), axis=1)

dataset_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
manifest = {
    "store": slug,
    "embedding_model": args.model,
    "provider": "gemini" if gemini else "openai",
    "dimensions": {"requested": dims, "actual": len(sample[0])},
    "normalization": {"applied_by_script": args.normalize},
    "chunking": {**RECIPE, "embedded_text": "chunk text with newlines replaced by spaces"},
    "id_scheme": "{doc_id}-{chunk_index}",
    "metadata_fields": ["title", "url", "source"],
    "collection_name": "ai_tutor_knowledge",
    "distance": "cosine",
    "source_dataset": {"filename": args.input.name, "sha256": dataset_sha, "documents": len(docs)},
    "counts": {"documents": len(docs), "chunks": len(records), "vectors": count},
    "tokens_embedded_cl100k": total_tokens,
    "estimated_cost_usd": round(est_cost, 4),
    "build_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(store_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

zip_path = shutil.make_archive(str(args.output / f"{slug}-{dataset_sha[:8]}"), "zip",
                               root_dir=args.output, base_dir=slug)
print(f"✅ {slug}: {count} vectors @ {len(sample[0])}d | "
      f"norms min/mean/max {norms.min():.4f}/{norms.mean():.4f}/{norms.max():.4f} | "
      f"{time.time() - t0:.0f}s | store {sum(f.stat().st_size for f in store_dir.rglob('*') if f.is_file()) / 1e6:.0f} MB "
      f"| zip {Path(zip_path).name} ({Path(zip_path).stat().st_size / 1e6:.0f} MB)")
