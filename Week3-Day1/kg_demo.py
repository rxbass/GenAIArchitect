from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

from dotenv import load_dotenv
import os


# ============================================
# LOAD ENVIRONMENT VARIABLES
# ============================================

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


print("\n--- Chunk Lengths ---")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}: {len(chunk.page_content)} characters")


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
# STEP 4: GRAPH STORAGE - NEO4J AURA
# ============================================

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE")
)

print("\n--- CONNECTED TO NEO4J AURA ---")


# Store graph documents in Neo4j Aura
graph.add_graph_documents(graph_documents)

print("--- GRAPH STORED IN NEO4J AURA ---")


# ============================================
# STEP 5: VIEW NEO4J GRAPH SCHEMA
# ============================================

print("\n--- NEO4J SCHEMA ---")

print(graph.schema)


# ============================================
# STEP 6: USER QUESTION
# ============================================

question = "What companies is Elon Musk the CEO of?"


# ============================================
# STEP 6.1: LLM GENERATES CYPHER
# ============================================

prompt = f"""
You are a Neo4j Cypher expert.

Generate a Cypher query to answer the user's question.

Use ONLY the following Neo4j schema:

{graph.schema}

Rules:
- Return only the Cypher query.
- Do not explain the query.
- Do not invent node labels, properties, or relationships.
- Use the exact property name `id`.
- Do not use Markdown code fences.

User question:
{question}
"""


response = llm.invoke(prompt)


# ============================================
# STEP 6.2: CLEAN LLM CYPHER OUTPUT
# ============================================

cypher_query = response.content.strip()


# Remove Markdown code fences if the LLM
# still returns them despite the prompt.
if cypher_query.startswith("```"):

    lines = cypher_query.splitlines()

    # Remove first line such as:
    # ```cypher
    lines = lines[1:]

    # Remove final line:
    # ```
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    cypher_query = "\n".join(lines).strip()


print("\n--- GENERATED CYPHER ---")
print(cypher_query)


# ============================================
# STEP 7: EXECUTE CYPHER IN NEO4J AURA
# ============================================

result = graph.query(cypher_query)


print("\n--- QUERY RESULT ---")
print(result)

# ============================================
# STEP 8: GENERATE FINAL RAG ANSWER
# ============================================

answer_prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information
retrieved from the Neo4j knowledge graph.

User question:
{question}

Retrieved graph information:
{result}

Rules:
- Use only the retrieved information.
- Do not invent or assume facts.
- Give a concise and clear answer.
- If the retrieved information is insufficient, say:
  "I don't have enough information in the knowledge graph."
"""

answer_response = llm.invoke(answer_prompt)

final_answer = answer_response.content.strip()

print("\n--- FINAL ANSWER ---")
print(final_answer)