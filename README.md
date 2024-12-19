# Semantic Text Embedding

## Overview
Semantic Text Embedding is a Python module designed to process text documents, divide them into semantically meaningful chunks, generate embeddings using a modern embedding model, and store them efficiently in a vector database. The module supports multiple file formats (PDF, DOCX) and implements various chunking strategies to cater to diverse use cases.

## Features
- **File Format Support**: Handles PDF and DOCX files seamlessly.
- **Chunking Strategies**:
  - Fixed-size chunks with overlap.
  - Sentence-based splitting.
  - Paragraph-based splitting.
- **Embedding Model**: Utilizes `SentenceTransformer` for generating high-quality embeddings.
- **Efficient Storage**: Stores embeddings in a SQLite database for easy retrieval.

## Requirements
- Python 3.8+
- Libraries:
  - `sentence-transformers`
  - `numpy`
  - `sqlalchemy`
  - `pdfplumber`
  - `python-docx`

Install the required libraries using:
```bash
pip install -r requirements.txt
```

## Installation
Clone the repository:
```bash
git clone https://github.com/buzagloidan/semantic-text-embedding.git
cd semantic-text-embedding
```

## Usage
Run the module with the following command:
```bash
python text_embedding_module.py <file_path> <chunking_strategy> [--db_path <db_path>]
```

### Arguments:
- `file_path`: Path to the input file (PDF or DOCX).
- `chunking_strategy`: The strategy for splitting the text. Options:
  - `fixed_overlap`: Fixed-size chunks with overlap.
  - `sentence_splitter`: Sentence-based splitting.
  - `paragraph_splitter`: Paragraph-based splitting.
- `--db_path` (optional): Path to the SQLite database. Default is `sqlite:///embeddings.db`.

### Example:
```bash
python text_embedding_module.py sample.pdf sentence_splitter
```

## How It Works
1. **File Reading**: Extracts text from PDF or DOCX files.
2. **Text Splitting**: Divides text into chunks using the specified strategy.
3. **Embedding Generation**: Uses `SentenceTransformer` to generate embeddings for each chunk.
4. **Storage**: Saves chunks and their embeddings to a SQLite database for future use.

## Database Schema
The embeddings are stored in a SQLite database with the following structure:

| Column     | Type       | Description                |
|------------|------------|----------------------------|
| `id`       | Integer    | Unique identifier          |
| `chunk`    | String     | Text chunk                 |
| `embedding`| Binary     | Binary representation of embedding |

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve this project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [SentenceTransformers](https://www.sbert.net/) for the embedding model.
- [SQLAlchemy](https://www.sqlalchemy.org/) for database management.
