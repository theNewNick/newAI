from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")  # "business4"

# Create a Pinecone instance
pc = Pinecone(api_key=api_key)

# Check if index exists, create if not
existing = pc.list_indexes().names()
if index_name not in existing:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",        # we know it's AWS
            region="us-east-1"  # from your console
        )
    )

index = pc.Index(index_name)
stats = index.describe_index_stats()
print("Stats:", stats)
