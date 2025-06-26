from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import shutil

# Import ctransformers for GGUF model loading
try:
    from ctransformers import AutoModelForCausalLM
    CTTRANSFORMERS_AVAILABLE = True
except ImportError:
    CTTRANSFORMERS_AVAILABLE = False
    print("⚠️  ctransformers not available. Install with: pip install ctransformers")

app = FastAPI(title="AI Mini Chat Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple conversation memory (LangChain-inspired)
class ConversationMemory:
    def __init__(self):
        self.messages = []
    
    def add_user_message(self, content):
        self.messages.append({"type": "user", "content": content})
    
    def add_ai_message(self, content):
        self.messages.append({"type": "ai", "content": content})
    
    def get_messages(self):
        return self.messages
    
    def clear(self):
        self.messages = []
    
    def get_context(self, limit=10):
        """Get recent messages for context"""
        return self.messages[-limit:] if self.messages else []

# Initialize memory
memory = ConversationMemory()

# Global LLM model instance
llm_model = None

# Model configuration
MODEL_CONFIGS = {
    "mythomax": {
        "name": "MythoMax L2 13B",
        "model_path": "mythomax-l2-13b.Q6_K.gguf",
        "description": "High-quality responses, good for general conversation",
        "model_type": "llama"
    },
    "deepseek": {
        "name": "DeepSeek",
        "model_path": None,
        "description": "Excellent for coding and technical questions",
        "model_type": None
    },
    "mistral": {
        "name": "Mistral",
        "model_path": None,
        "description": "Balanced performance across various tasks",
        "model_type": None
    },
    "openchat": {
        "name": "OpenChat",
        "model_path": None,
        "description": "Good for creative writing and casual conversation",
        "model_type": None
    }
}

def load_gguf_model():
    """Load the GGUF model"""
    global llm_model
    
    if not CTTRANSFORMERS_AVAILABLE:
        print("❌ ctransformers not available. Using fallback responses.")
        return False
    
    try:
        model_path = MODEL_CONFIGS["mythomax"]["model_path"]
        model_type = MODEL_CONFIGS["mythomax"]["model_type"]
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"🔄 Loading GGUF model: {model_path}")
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type=model_type,
            gpu_layers=0,  # Set to >0 if you have GPU
            max_new_tokens=64,  # Further reduced for faster responses
            temperature=0.6,  # Balanced for speed and quality
            top_p=0.8,  # Further reduced for faster generation
            repetition_penalty=1.01,  # Minimal penalty for speed
            threads=12  # Increased threads for better performance
        )
        print("✅ GGUF model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading GGUF model: {e}")
        return False

def get_llm_response(prompt, context=None):
    """Get response from LLM (GGUF model or fallback)"""
    global llm_model
    
    if llm_model is not None:
        try:
            # Build context-aware prompt (limit context for speed)
            if context:
                # Only use last 2 messages for context to improve speed
                context_text = "\n".join([f"{msg['type']}: {msg['content']}" for msg in context[-2:]])
                full_prompt = f"Context:\n{context_text}\n\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\nAssistant:"
            
            # Limit prompt length for faster processing
            if len(full_prompt) > 1000:
                full_prompt = full_prompt[-1000:] + "\nAssistant:"
            
            # Generate response
            response = llm_model(full_prompt)
            
            # Clean up response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            # Limit response length for faster display
            if len(response) > 500:
                response = response[:500] + "..."
            
            return response if response else "I understand your message. How can I help you further?"
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return get_fallback_response(prompt)
    else:
        return get_fallback_response(prompt)

def get_fallback_response(prompt):
    """Fallback response when GGUF model is not available"""
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ["hello", "hi", "hey", "hii"]):
        return "Hello! I'm your AI assistant. How can I help you today?"
    elif "how are you" in prompt_lower:
        return "I'm doing great, thank you for asking! I'm ready to assist you."
    elif "what can you do" in prompt_lower or "capabilities" in prompt_lower:
        return "I can help you with conversations, answer questions, process documents, and assist with various tasks. I use conversation memory to maintain context across our chat."
    elif "bye" in prompt_lower or "goodbye" in prompt_lower:
        return "Goodbye! It was nice chatting with you. Have a great day!"
    elif "help" in prompt_lower:
        return "I'm here to help! You can ask me questions, have conversations, or upload documents. I maintain context throughout our conversation."
    elif "weather" in prompt_lower:
        return "I can't check the weather directly, but I can help you with other questions and tasks!"
    elif "time" in prompt_lower:
        return "I don't have access to real-time information, but I can help you with other questions and tasks!"
    elif "joke" in prompt_lower or "funny" in prompt_lower:
        return "Why don't scientists trust atoms? Because they make up everything! 😄"
    elif "thank" in prompt_lower:
        return "You're welcome! I'm happy to help. Is there anything else you'd like to know?"
    elif "memory" in prompt_lower or "remember" in prompt_lower:
        msg_count = len(memory.get_messages())
        return f"I remember our conversation! We've exchanged {msg_count} messages so far. I maintain context throughout our chat."
    else:
        return f"I understand you said: '{prompt}'. I'm here to help with your questions and tasks!"

def process_document_simple(file_path: str, file_type: str):
    """Simple document processing (LangChain-inspired)"""
    try:
        # Read document content
        if file_type == "text/plain":
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            # For other file types, we'll need additional libraries
            content = f"Document content from {os.path.basename(file_path)}"
        
        # Simple text processing (LangChain-inspired chunking)
        lines = content.split('\n')
        word_count = len(content.split())
        
        # Simple text chunking (like LangChain's text splitter)
        chunks = []
        chunk_size = 1000
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return {
            "success": True,
            "filename": os.path.basename(file_path),
            "word_count": word_count,
            "line_count": len(lines),
            "chunks": len(chunks),
            "content_preview": content[:200] + "..." if len(content) > 200 else content
        }
            
    except Exception as e:
        print(f"Error processing document: {e}")
        return {"success": False, "error": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    print("🚀 Starting AI Chatbot Backend...")
    print("📚 Available models:", list(MODEL_CONFIGS.keys()))
    
    # Load GGUF model
    if load_gguf_model():
        print("✅ GGUF model loaded successfully!")
    else:
        print("⚠️  Using fallback responses (no GGUF model)")
    
    print("✅ LangChain-inspired components initialized successfully")
    print("🌐 Backend will be available at: http://localhost:8000")
    print("📖 API docs will be available at: http://localhost:8000/docs")

@app.get("/")
async def root():
    return {
        "message": "AI Chatbot Backend is running!",
        "version": "1.0.0",
        "langchain_available": True,
        "gguf_model_loaded": llm_model is not None,
        "models": list(MODEL_CONFIGS.keys()),
        "features": [
            "Conversation memory (LangChain-inspired)",
            "Document processing",
            "Multiple model support",
            "RESTful API",
            "Context-aware responses",
            "GGUF model support" if llm_model else "Fallback responses"
        ]
    }

@app.get("/models")
async def get_models():
    """Get available models"""
    return {"models": MODEL_CONFIGS}

@app.post("/chat")
async def chat(
    message: str = Form(...),
    model: str = Form("mythomax"),
    history: str = Form("[]")
):
    """Chat endpoint with LangChain-inspired architecture"""
    try:
        # Parse history
        try:
            history_list = json.loads(history)
        except:
            history_list = []
        
        print(f"Received message: {message} (Model: {model})")
        
        # Use LangChain-inspired memory
        try:
            # Add user message to memory
            memory.add_user_message(message)
            
            # Get context for better responses
            context = memory.get_context()
            
            # Get response using simple LLM
            response = get_llm_response(message, context)
            
            # Add AI response to memory
            memory.add_ai_message(response)
            
            print("✅ LangChain-inspired memory updated successfully")
            
        except Exception as e:
            print(f"Error in LangChain-inspired processing: {e}")
            # Fallback response
            response = get_fallback_response(message)
        
        # Add to history
        history_list.append([message, response])
        
        return JSONResponse({
            "response": response,
            "history": history_list,
            "model": model,
            "langchain_used": True,
            "memory_size": len(memory.get_messages())
        })
        
    except Exception as e:
        print("Error in /chat endpoint:", e)
        return JSONResponse({"response": f"Error: {e}", "error": True})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process document"""
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        result = process_document_simple(file_path, file.content_type)
        
        # Clean up
        os.remove(file_path)
        
        if result["success"]:
            return JSONResponse({
                "message": f"Document '{file.filename}' uploaded and processed successfully!",
                "filename": file.filename,
                "status": "success",
                "details": result
            })
        else:
            return JSONResponse({
                "message": f"Failed to process document '{file.filename}': {result.get('error', 'Unknown error')}",
                "filename": file.filename,
                "status": "error"
            })
            
    except Exception as e:
        print(f"Error in upload: {e}")
        return JSONResponse({
            "message": f"Upload error: {str(e)}",
            "status": "error"
        })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "langchain_available": True,
        "memory_size": len(memory.get_messages()),
        "models_available": len(MODEL_CONFIGS)
    }

@app.get("/memory")
async def get_memory():
    """Get conversation memory"""
    try:
        messages = memory.get_messages()
        return {
            "memory_size": len(messages),
            "messages": messages
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/clear-memory")
async def clear_memory():
    """Clear conversation memory"""
    try:
        memory.clear()
        return {"message": "Memory cleared successfully"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting AI Chatbot Backend with LangChain-inspired architecture...")
    print("Backend will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)