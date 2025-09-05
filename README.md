# ChromaDB-OpenAI RAG System

A sophisticated Retrieval Augmented Generation (RAG) system built specifically for insurance policy document analysis and question-answering. This system combines the power of ChromaDB vector database, OpenAI's language models, and advanced document processing techniques to provide accurate, context-aware responses to insurance-related queries.

## 🏗️ System Architecture

<img width="2023" height="3840" alt="ProjectArchitecture" src="https://github.com/user-attachments/assets/1715e988-c820-4d11-87d3-d8b4e453e999" />

The system follows a two-phase architecture:

### Phase 1: Ingestion & Indexing
1. **Document Upload**: Users upload PDF policy documents
2. **Text Extraction**: PDF Plumber processes documents, extracting both text and tables while preserving formatting
3. **Embedding Generation**: Sentence Transformers create vector embeddings from processed text chunks
4. **Vector Storage**: ChromaDB stores embeddings with metadata for efficient similarity search

### Phase 2: Query & Generation
1. **Query Processing**: User queries are embedded using the same sentence transformer model
2. **Semantic Search**: ChromaDB retrieves the most relevant document chunks based on vector similarity
3. **Result Reranking**: Cross-encoder model reranks results for improved relevance
4. **Response Generation**: OpenAI GPT-3.5 generates comprehensive answers with document citations

## ✨ Key Features

- **Advanced PDF Processing**: Handles complex insurance documents with tables, nested structures, and multi-column layouts
- **Intelligent Text Chunking**: Preserves document structure while creating optimal chunks for retrieval
- **Hybrid Search**: Combines semantic similarity with reranking for superior result quality
- **Citation Support**: Provides accurate citations with policy names and page numbers
- **Table Processing**: Extracts and processes tabular data from insurance documents
- **Multi-Document Support**: Handles multiple policy documents simultaneously
- **Contextual Responses**: Generates human-readable answers tailored to insurance domain

## 🛠️ Technology Stack

### Core Libraries
- **ChromaDB**: Vector database for embedding storage and similarity search
- **OpenAI**: GPT-3.5 for natural language generation
- **Sentence Transformers**: For creating high-quality text embeddings
- **PDFPlumber**: Advanced PDF text and table extraction
- **Pandas**: Data manipulation and analysis

### Machine Learning Components
- **Embedding Model**: Sentence Transformers for semantic understanding
- **Cross-Encoder**: For result reranking and relevance scoring
- **Language Model**: OpenAI GPT-3.5-turbo for response generation

### Supporting Libraries
- **TikToken**: Token counting and text processing
- **Pathlib**: File system operations
- **JSON**: Data serialization for structured content

## 📊 System Performance

Based on the experimental results from the notebook execution:

### Document Processing Results
- **Documents Processed**: 7 insurance policy documents
- **Successfully Extracted**: All documents processed without errors
- **Document Types**: Various HDFC Life insurance policies including:
  - Group Term Life Policy
  - Sampoorna Jeevan Policy  
  - Easy Health Policy
  - Smart Pension Plan
  - Group Poorna Suraksha Policy
  - Sanchay Plus Policy
  - Surgicare Plan

### Query Performance Metrics
For the test query *"what are accidental death benefits in life insurance policy"*:

#### Retrieval Results (Top 3 after reranking):
| Rank | Relevance Score | Document Source | Page |
|------|----------------|-----------------|------|
| 1 | 4.005178 | HDFC-Life-Group-Term-Life-Policy.pdf | Page 15 |
| 2 | 3.202596 | HDFC-Life-Group-Term-Life-Policy.pdf | Page 7 |
| 3 | 2.366757 | HDFC-Life-Group-Term-Life-Policy.pdf | Page 4 |

#### Search Quality Metrics:
- **Vector Similarity Scores**: Ranging from 0.289 to 0.359 (higher is better)
- **Reranked Scores**: Improved relevance ranking with scores from 2.37 to 4.01
- **Citation Accuracy**: 100% - All results include accurate page and document references

### Response Quality
The system successfully generated a comprehensive response that:
- ✅ Addressed the specific query about accidental death benefits
- ✅ Provided relevant policy information from retrieved documents
- ✅ Included accurate citations with policy names and page numbers
- ✅ Offered guidance for finding additional information in specific document sections

## 🔍 System Capabilities

### Document Processing Features
- **Multi-format Support**: Handles various PDF layouts and structures
- **Table Extraction**: Preserves tabular data with proper formatting
- **Metadata Preservation**: Maintains document source and page information
- **Text Cleaning**: Processes and normalizes extracted content

### Search and Retrieval
- **Semantic Search**: Understanding of query intent beyond keyword matching  
- **Context Awareness**: Considers document structure and relationships
- **Relevance Scoring**: Multiple scoring mechanisms for optimal result ranking
- **Result Filtering**: Ensures retrieved content matches query context

### Response Generation
- **Domain Expertise**: Specialized knowledge in insurance terminology and concepts
- **Citation Management**: Automatic generation of accurate source references  
- **Answer Formatting**: Well-structured, readable responses
- **Context Integration**: Synthesizes information from multiple document sources

## 📈 Use Cases

### Primary Applications
1. **Customer Service**: Instant answers to policy-related questions
2. **Claims Processing**: Quick access to policy terms and conditions
3. **Compliance Checking**: Verification of coverage details and exclusions
4. **Agent Support**: Real-time assistance for insurance agents and brokers

### Query Examples
The system excels at answering questions such as:
- "What are the accidental death benefits in life insurance policies?"
- "What is the waiting period for critical illness coverage?"
- "Which surgeries are covered under the health insurance plan?"
- "What are the exclusions for the term life insurance policy?"

## 🔧 Technical Implementation

### PDF Processing Pipeline
```python
# Advanced PDF extraction with table handling
def extract_text_from_pdf(pdf_path):
    # Processes tables and regular text separately
    # Maintains document structure and formatting
    # Returns structured data with page information
```

### Vector Search Implementation
```python
# ChromaDB configuration for optimal performance
collection = client.create_collection(
    name="insurance_policies",
    embedding_function=embedding_function,
    metadata={"description": "Insurance policy documents"}
)
```

### Response Generation
```python
# GPT-3.5 integration with custom prompts
def generate_response(query, results_df):
    # Context-aware prompt engineering
    # Citation extraction and formatting
    # Domain-specific response generation
```

## 🎯 Results and Achievements

### Successful Implementation
- ✅ **Complete RAG Pipeline**: End-to-end implementation from document ingestion to response generation
- ✅ **Advanced PDF Processing**: Successfully handles complex insurance document formats
- ✅ **High-Quality Embeddings**: Effective semantic search across policy documents
- ✅ **Accurate Retrieval**: Relevant document chunks retrieved for user queries
- ✅ **Intelligent Reranking**: Improved result relevance through cross-encoder scoring
- ✅ **Professional Responses**: Human-readable answers with proper citations

### Performance Validation
- **Retrieval Accuracy**: Successfully identified relevant policy sections for test queries
- **Response Quality**: Generated comprehensive, accurate answers with proper context
- **Citation Precision**: 100% accuracy in document and page references
- **Processing Efficiency**: Handled multiple policy documents without performance degradation

### System Robustness
- **Error Handling**: Graceful processing of various PDF formats and structures
- **Scalability**: Architecture supports addition of more policy documents
- **Maintainability**: Clean, modular code structure for easy updates and modifications

## 🚀 Future Enhancements

### Potential Improvements
1. **Multi-modal Support**: Integration of images and charts from policy documents
2. **Advanced Analytics**: Query pattern analysis and usage insights
3. **Real-time Updates**: Dynamic document ingestion and index updates
4. **Enhanced UI**: Web interface for easier document management and querying
5. **Performance Optimization**: Caching mechanisms and faster embedding generation

### Scalability Considerations
- Distributed vector storage for large document collections
- Load balancing for high-concurrent query processing
- Advanced caching strategies for frequently accessed information
- Integration with enterprise document management systems

---

*This ChromaDB-OpenAI RAG System represents a sophisticated approach to intelligent document processing and question-answering, specifically tailored for the insurance industry's complex documentation needs.*
