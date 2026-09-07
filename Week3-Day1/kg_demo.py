from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================
# STEP 1: DOCUMENT LOADER
# ============================================

loader = TextLoader("sample.txt")

documents = loader.load()

print("Number of documents:", len(documents))

print("\n--- Document Content ---")
print(documents[0].page_content)

print("\n--- Document Metadata ---")
print(documents[0].metadata)


# ============================================
# STEP 2: TEXT SPLITTER
# ============================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("\n--- TEXT SPLITTING ---")
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)


# ============================================
# STEP 3: LLM GRAPH TRANSFORMER
# ============================================

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

llm_transformer = LLMGraphTransformer(
    llm=llm
)

graph_documents = llm_transformer.convert_to_graph_documents(chunks)


print("\n--- GRAPH DOCUMENTS ---")

for i, graph_doc in enumerate(graph_documents):

    print(f"\nGraph Document {i + 1}")

    print("\nNodes:")

    for node in graph_doc.nodes:
        print(f"  {node.id} -> {node.type}")

    print("\nRelationships:")

    for relationship in graph_doc.relationships:
        print(
            f"  {relationship.source.id}"
            f" --[{relationship.type}]--> "
            f"{relationship.target.id}"
        )

# ============================================
# STEP 4: GRAPH STORAGE - NEO4J
# ============================================

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE")
)

print("\n--- CONNECTED TO NEO4J ---")


# Store graph documents in Neo4j
graph.add_graph_documents(graph_documents)

print("--- GRAPH STORED IN NEO4J ---")