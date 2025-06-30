# AI Mini Chat

An advanced AI chatbot using LangChain with free, open-source LLMs (DeepSeek, Mistral, OpenChat) and a modern tech stack including Ollama for local model hosting, ChromaDB for vector storage, HuggingFace embeddings, FastAPI backend, and Flask frontend.

## Features

- 🤖 **Multiple AI Models**: Support for MythoMax L2 13B, DeepSeek, Mistral, and OpenChat
- 🧠 **Local Processing**: Run AI models locally using Ollama or direct GGUF loading
- 📚 **Document Upload**: Upload and chat about documents (PDF, DOC, DOCX, TXT)
- 🔍 **Vector Search**: ChromaDB-powered document retrieval and context-aware responses
- 🌐 **Modern UI**: Beautiful, responsive web interface with glassmorphism design
- ⚡ **Fast Performance**: Optimized for quick responses and smooth interactions
- 🔒 **Privacy-First**: All processing happens locally on your machine

## Tech Stack

- **Backend**: FastAPI with Python 3.13
- **Frontend**: Flask with modern HTML/CSS/JavaScript
- **AI Models**: Ollama + GGUF models (MythoMax L2 13B, DeepSeek, Mistral, OpenChat)
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace Transformers
- **Language Chain**: LangChain for conversation management

## Quick Start

### Prerequisites

- Python 3.13 or higher
- Windows 10/11 (tested on Windows 10.0.26100)
- At least 8GB RAM (16GB recommended for larger models)
- Ollama installed (optional, for additional models)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd aiminichat
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```bash
   # Install backend dependencies
   cd backend
   pip install -r requirements.txt
   cd ..
   
   # Install frontend dependencies
   cd frontend
   pip install -r requirements.txt
   cd ..
   ```

4. **Download AI Model** (if not using Ollama):
   - Place your GGUF model file in the `backend/` directory
   - Update the model path in `backend/main.py` if needed

### Running the Application

#### Option 1: Using the Startup Script (Recommended)
```bash
# Simply run the batch file
start_services.bat
```

#### Option 2: Manual Startup

1. **Start the Backend**:
   ```bash
   cd backend
   python main.py
   ```
   The backend will be available at: http://localhost:8000

2. **Start the Frontend** (in a new terminal):
   ```bash
   cd frontend
   python simple_app.py
   ```
   The frontend will be available at: http://localhost:5000

3. **Open your browser** and navigate to: http://localhost:5000

## Usage

### Basic Chat
1. Select your preferred AI model from the dropdown
2. Type your message in the text area
3. Press Enter or click Send
4. The AI will respond with context-aware answers

### Document Upload
1. Click "Upload Document" to select a file
2. Supported formats: PDF, DOC, DOCX, TXT
3. The document will be processed and added to the knowledge base
4. Ask questions about the uploaded document

### Model Selection
- **MythoMax L2 13B**: High-quality responses, good for general conversation
- **DeepSeek**: Excellent for coding and technical questions
- **Mistral**: Balanced performance across various tasks
- **OpenChat**: Good for creative writing and casual conversation

## Configuration

### Backend Configuration
Edit `backend/main.py` to customize:
- Model paths and parameters
- ChromaDB settings
- API endpoints

### Frontend Configuration
Edit `frontend/simple_app.py` to customize:
- Backend API URL
- Port settings
- UI behavior

## Troubleshooting

### Common Issues

1. **Port Already in Use**:
   - Backend: Change port in `backend/main.py`
   - Frontend: Change port in `frontend/simple_app.py`

2. **Model Not Found**:
   - Ensure the GGUF model file is in the correct location
   - Check the model path in `backend/main.py`

3. **Import Errors**:
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

4. **Memory Issues**:
   - Use smaller models for limited RAM
   - Close other applications to free memory

### Performance Tips

- Use SSD storage for faster model loading
- Increase system RAM for better performance
- Close unnecessary applications while running
- Use quantized models (Q4_K, Q6_K) for faster inference

## API Documentation

Once the backend is running, visit:
- **API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Main Endpoints

- `GET /`: Health check
- `POST /chat`: Send a message and get AI response
- `POST /upload`: Upload a document for processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request


## Acknowledgments

- LangChain for the conversation framework
- Ollama for local model hosting
- ChromaDB for vector storage
- HuggingFace for embeddings and models
- The open-source AI community for amazing models and tools 
