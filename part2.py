import os
import re
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import docx
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class EmbeddingEntry(Base):
    __tablename__ = 'embeddings'
    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=False)

def read_pdf(file_path: str) -> str:
    with pdfplumber.open(file_path) as pdf:
        return ' '.join(page.extract_text() for page in pdf.pages if page.extract_text())

def read_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return '\n'.join([p.text for p in doc.paragraphs])

def split_text_fixed_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = [
        ' '.join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]
    return chunks

def split_text_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences

def split_text_paragraphs(text: str) -> List[str]:
    paragraphs = text.split('\n')
    return [p.strip() for p in paragraphs if p.strip()]

def create_embeddings(chunks: List[str], model: SentenceTransformer) -> List[np.ndarray]:
    return model.encode(chunks)

def save_embeddings_to_db(chunks: List[str], embeddings: List[np.ndarray], session):
    for chunk, embedding in zip(chunks, embeddings):
        entry = EmbeddingEntry(
            chunk=chunk,
            embedding=np.array(embedding).tobytes()
        )
        session.add(entry)
    session.commit()

def read_embeddings_from_db(db_path: str):
    # Initialize database
    engine = create_engine(db_path)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Query all embeddings
    entries = session.query(EmbeddingEntry).all()

    # Print the results
    for entry in entries:
        print(f"ID: {entry.id}, Chunk: {entry.chunk}, Embedding: {np.frombuffer(entry.embedding, dtype=np.float32)}")

    session.close()

def main(file_path: str = 'test.docx', db_path: str = 'sqlite:///embeddings.db'):
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Read file
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.pdf':
        text = read_pdf(file_path)
    elif ext == '.docx':
        text = read_docx(file_path)
    else:
        raise ValueError("Unsupported file format")

    # Define chunking strategies
    chunking_strategies = ['fixed_overlap', 'sentence_splitter', 'paragraph_splitter']

    for chunking_strategy in chunking_strategies:
        # Split text
        if chunking_strategy == 'fixed_overlap':
            chunks = split_text_fixed_overlap(text, chunk_size=100, overlap=20)
        elif chunking_strategy == 'sentence_splitter':
            chunks = split_text_sentences(text)
        elif chunking_strategy == 'paragraph_splitter':
            chunks = split_text_paragraphs(text)
        else:
            raise ValueError("Invalid chunking strategy")

        # Create embeddings
        embeddings = create_embeddings(chunks, model)

        # Initialize database
        engine = create_engine(db_path)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Save to database
        save_embeddings_to_db(chunks, embeddings, session)
        print(f"Embeddings saved to database for strategy: {chunking_strategy}.")

if __name__ == "__main__":
    main()  # No arguments needed
    read_embeddings_from_db('sqlite:///embeddings.db')  # Call the function to read embeddings
