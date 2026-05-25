import os
import glob
import argparse
from typing import List, Dict, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector

# Import shared components
try:
    from .Retriever import MiniLMEmbeddings
except ImportError:
    # Fallback for running as a script
    try:
        from Retriever import MiniLMEmbeddings
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from Retriever import MiniLMEmbeddings

class KnowledgeIngestor:
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        index_name: str = "essay_chunkAgentSpace",
        fulltext_index_name: str = "chunk_index",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        load_dotenv()
        self.neo4j_uri = neo4j_uri or os.environ.get("NEO4J_URI")
        self.neo4j_username = neo4j_username or os.environ.get("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.environ.get("NEO4J_PASSWORD")
        self.index_name = index_name
        self.fulltext_index_name = fulltext_index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embeddings = MiniLMEmbeddings(embedding_model_name)
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )

    def load_pdfs(self, path: str) -> List:
        docs = []
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, "*.pdf"))
        else:
            files = [path]
            
        for file in files:
            if not os.path.exists(file):
                print(f"File not found: {file}")
                continue
            loader = PyMuPDFLoader(file)
            docs.extend(loader.load())
        return docs

    def split_documents(self, docs: List) -> List:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        for i, c in enumerate(chunks):
            c.metadata["chunk_id"] = i
        return chunks

    def get_or_create_vectorstore(self):
        with self.driver.session() as session:
            result = session.run("""
                SHOW INDEXES
                YIELD name, type
                WHERE name = $index_name AND type = 'VECTOR'
                RETURN count(*) > 0 as exists
            """, index_name=self.index_name)
            
            record = result.single()
            index_exists = record['exists'] if record else False

        if index_exists:
            print(f"Connecting to existing vector index: {self.index_name}")
            vectorstore = Neo4jVector(
                embedding=self.embeddings,
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name=self.index_name,
                node_label="Chunk",
                text_node_property="text",
                embedding_node_property="embedding",
            )
        else:
            print(f"Creating new vector index: {self.index_name}")
            vectorstore = Neo4jVector.from_documents(
                documents=[],
                embedding=self.embeddings,
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name=self.index_name,
                node_label="Chunk",
                text_node_property="text",
                embedding_node_property="embedding",
            )
        return vectorstore

    def build_graph_relationships(self, chunks: List):
        by_source: Dict[str, List] = {}
        for c in chunks:
            src = c.metadata.get("source", "unknown")
            by_source.setdefault(src, []).append(c)

        with self.driver.session() as session:
            # Create fulltext index if not exists
            session.run(f"CREATE FULLTEXT INDEX {self.fulltext_index_name} IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]")
            
            for source, source_chunks in by_source.items():
                source_chunks.sort(key=lambda x: x.metadata["chunk_id"])

                session.run(
                    "MERGE (d:Document {name: $name}) SET d.chunk_count = $n",
                    {"name": source, "n": len(source_chunks)}
                )

                for i, c in enumerate(source_chunks):
                    session.run(
                        """
                        MATCH (d:Document {name: $source})
                        MATCH (c:Chunk {chunk_id: $cid})
                        MERGE (c)-[:PART_OF]->(d)
                        """,
                        {"source": source, "cid": c.metadata["chunk_id"]}
                    )

                    if i < len(source_chunks) - 1:
                        session.run(
                            """
                            MATCH (c1:Chunk {chunk_id: $c1})
                            MATCH (c2:Chunk {chunk_id: $c2})
                            MERGE (c1)-[:NEXT]->(c2)
                            """,
                            {
                                "c1": c.metadata["chunk_id"],
                                "c2": source_chunks[i + 1].metadata["chunk_id"],
                            },
                        )

    def ingest(self, path: str):
        print(f"Starting ingestion for: {path}")
        pdfs = self.load_pdfs(path)
        if not pdfs:
            print("No documents loaded. Aborting.")
            return
            
        chunks = self.split_documents(pdfs)
        print(f"Split into {len(chunks)} chunks.")
        
        vectorstore = self.get_or_create_vectorstore()
        vectorstore.add_documents(chunks)
        print("Added documents to vector store.")
        
        self.build_graph_relationships(chunks)
        print("Built graph relationships.")
        print("Ingestion complete.")

def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Neo4j Knowledge Graph")
    parser.add_argument("path", help="Path to a PDF file or a directory containing PDFs")
    parser.add_argument("--index", default="essay_chunkAgentSpace", help="Neo4j vector index name")
    parser.add_argument("--fulltext", default="chunk_index", help="Neo4j fulltext index name")
    parser.add_argument("--size", type=int, default=500, help="Chunk size")
    parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap")
    
    args = parser.parse_args()
    
    ingestor = KnowledgeIngestor(
        index_name=args.index,
        fulltext_index_name=args.fulltext,
        chunk_size=args.size,
        chunk_overlap=args.overlap
    )
    ingestor.ingest(args.path)

if __name__ == "__main__":
    main()
