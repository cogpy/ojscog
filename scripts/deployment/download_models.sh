#!/bin/bash
#
# Model Download and Setup Script
# Downloads and configures AI models for the autonomous agents system
#
# Usage: ./download_models.sh [--model-type all|llm|vision|speech] [--size small|medium|large]
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"
MODEL_TYPE="all"
MODEL_SIZE="medium"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--model-type all|llm|vision|speech] [--size small|medium|large]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create models directory
create_model_dirs() {
    log_info "Creating model directories..."
    
    mkdir -p "$MODELS_DIR/llm"
    mkdir -p "$MODELS_DIR/vision"
    mkdir -p "$MODELS_DIR/speech"
    mkdir -p "$MODELS_DIR/embeddings"
    mkdir -p "$MODELS_DIR/tokenizers"
    
    log_success "Model directories created"
}

# Download LLM models
download_llm_models() {
    log_info "Downloading LLM models (size: $MODEL_SIZE)..."
    
    cd "$MODELS_DIR/llm"
    
    case $MODEL_SIZE in
        small)
            # LLaMA 2 7B Q4_0 (~3.5GB)
            MODEL_URL="https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf"
            MODEL_FILE="llama-7b-q4.gguf"
            ;;
        medium)
            # LLaMA 2 7B Q5_K_M (~4.8GB)
            MODEL_URL="https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q5_K_M.gguf"
            MODEL_FILE="llama-7b-q5.gguf"
            ;;
        large)
            # LLaMA 2 13B Q4_0 (~7GB)
            MODEL_URL="https://huggingface.co/TheBloke/Llama-2-13B-GGUF/resolve/main/llama-2-13b.Q4_0.gguf"
            MODEL_FILE="llama-13b-q4.gguf"
            ;;
    esac
    
    if [ -f "$MODEL_FILE" ]; then
        log_info "LLM model already exists: $MODEL_FILE"
    else
        log_info "Downloading LLM model from HuggingFace..."
        log_info "URL: $MODEL_URL"
        
        wget -c "$MODEL_URL" -O "$MODEL_FILE.tmp"
        mv "$MODEL_FILE.tmp" "$MODEL_FILE"
        
        log_success "LLM model downloaded: $MODEL_FILE"
    fi
    
    # Download Mistral 7B as alternative
    MISTRAL_FILE="mistral-7b-q4.gguf"
    if [ ! -f "$MISTRAL_FILE" ]; then
        log_info "Downloading Mistral 7B model..."
        MISTRAL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_0.gguf"
        
        wget -c "$MISTRAL_URL" -O "$MISTRAL_FILE.tmp"
        mv "$MISTRAL_FILE.tmp" "$MISTRAL_FILE"
        
        log_success "Mistral model downloaded: $MISTRAL_FILE"
    fi
    
    # Create symlink to default model
    ln -sf "$MODEL_FILE" "$MODELS_DIR/llama-7b-q4.gguf"
    
    log_success "LLM models setup complete"
}

# Download embedding models
download_embedding_models() {
    log_info "Downloading embedding models..."
    
    cd "$MODELS_DIR/embeddings"
    
    # Download all-MiniLM-L6-v2 for embeddings
    EMBEDDING_MODEL="all-MiniLM-L6-v2"
    
    if [ ! -d "$EMBEDDING_MODEL" ]; then
        log_info "Downloading sentence-transformers model..."
        
        # Use Python to download via transformers
        python3.11 << EOF
from sentence_transformers import SentenceTransformer
import os

model_path = os.path.join('$MODELS_DIR', 'embeddings', '$EMBEDDING_MODEL')
model = SentenceTransformer('sentence-transformers/$EMBEDDING_MODEL')
model.save(model_path)
print(f"Model saved to {model_path}")
EOF
        
        log_success "Embedding model downloaded: $EMBEDDING_MODEL"
    else
        log_info "Embedding model already exists: $EMBEDDING_MODEL"
    fi
}

# Download vision models
download_vision_models() {
    log_info "Downloading vision models..."
    
    cd "$MODELS_DIR/vision"
    
    # Download Stable Diffusion model (if needed)
    if [ "$MODEL_SIZE" == "large" ]; then
        log_info "Downloading Stable Diffusion model..."
        
        SD_MODEL="stable-diffusion-v1-5"
        if [ ! -d "$SD_MODEL" ]; then
            python3.11 << EOF
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.save_pretrained('$MODELS_DIR/vision/$SD_MODEL')
print("Stable Diffusion model downloaded")
EOF
            
            log_success "Stable Diffusion model downloaded"
        else
            log_info "Stable Diffusion model already exists"
        fi
    fi
    
    # Download NCNN models for mobile inference
    log_info "Downloading NCNN vision models..."
    
    NCNN_MODELS_URL="https://github.com/nihui/ncnn-assets/releases/download/20230223"
    MODELS=("yolov5s" "mobilenet_v2" "squeezenet")
    
    for model in "${MODELS[@]}"; do
        if [ ! -f "${model}.param" ]; then
            wget -c "${NCNN_MODELS_URL}/${model}.param" || log_warning "Failed to download ${model}.param"
            wget -c "${NCNN_MODELS_URL}/${model}.bin" || log_warning "Failed to download ${model}.bin"
        fi
    done
    
    log_success "Vision models setup complete"
}

# Download speech models
download_speech_models() {
    log_info "Downloading speech models..."
    
    cd "$MODELS_DIR/speech"
    
    # Download Piper TTS voices
    log_info "Downloading Piper TTS voices..."
    
    PIPER_VOICES_URL="https://github.com/rhasspy/piper/releases/download/v1.2.0"
    VOICES=("en_US-lessac-medium" "en_US-amy-medium")
    
    for voice in "${VOICES[@]}"; do
        if [ ! -f "${voice}.onnx" ]; then
            log_info "Downloading voice: $voice"
            wget -c "${PIPER_VOICES_URL}/${voice}.onnx" || log_warning "Failed to download $voice"
            wget -c "${PIPER_VOICES_URL}/${voice}.onnx.json" || log_warning "Failed to download ${voice} config"
        fi
    done
    
    # Download Sherpa-ONNX STT models
    log_info "Downloading Sherpa-ONNX STT models..."
    
    SHERPA_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
    STT_MODEL="sherpa-onnx-streaming-zipformer-en-2023-06-26"
    
    if [ ! -d "$STT_MODEL" ]; then
        log_info "Downloading STT model..."
        wget -c "${SHERPA_URL}/${STT_MODEL}.tar.bz2"
        tar -xjf "${STT_MODEL}.tar.bz2"
        rm "${STT_MODEL}.tar.bz2"
        log_success "STT model downloaded: $STT_MODEL"
    else
        log_info "STT model already exists: $STT_MODEL"
    fi
    
    log_success "Speech models setup complete"
}

# Download tokenizers
download_tokenizers() {
    log_info "Downloading tokenizers..."
    
    cd "$MODELS_DIR/tokenizers"
    
    # Download sentencepiece tokenizer for LLaMA
    TOKENIZER_URL="https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model"
    
    if [ ! -f "tokenizer.model" ]; then
        wget -c "$TOKENIZER_URL"
        log_success "Tokenizer downloaded"
    else
        log_info "Tokenizer already exists"
    fi
}

# Verify downloads
verify_models() {
    log_info "Verifying model downloads..."
    
    ERRORS=0
    
    # Check LLM models
    if [ ! -f "$MODELS_DIR/llama-7b-q4.gguf" ]; then
        log_error "LLM model not found"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check embeddings
    if [ ! -d "$MODELS_DIR/embeddings/all-MiniLM-L6-v2" ]; then
        log_warning "Embedding model not found"
    fi
    
    if [ $ERRORS -eq 0 ]; then
        log_success "All critical models verified"
    else
        log_error "$ERRORS critical models missing"
        exit 1
    fi
}

# Create model registry
create_model_registry() {
    log_info "Creating model registry..."
    
    REGISTRY_FILE="$MODELS_DIR/model_registry.json"
    
    cat > "$REGISTRY_FILE" << EOF
{
  "version": "1.0.0",
  "updated": "$(date -Iseconds)",
  "models": {
    "llm": {
      "primary": {
        "name": "LLaMA 2 7B",
        "path": "$MODELS_DIR/llama-7b-q4.gguf",
        "type": "gguf",
        "quantization": "q4_0",
        "size_gb": $(du -sh "$MODELS_DIR/llama-7b-q4.gguf" 2>/dev/null | cut -f1 || echo "unknown"),
        "context_length": 4096
      },
      "alternatives": [
        {
          "name": "Mistral 7B",
          "path": "$MODELS_DIR/llm/mistral-7b-q4.gguf",
          "type": "gguf",
          "quantization": "q4_0"
        }
      ]
    },
    "embeddings": {
      "primary": {
        "name": "all-MiniLM-L6-v2",
        "path": "$MODELS_DIR/embeddings/all-MiniLM-L6-v2",
        "dimensions": 384,
        "max_seq_length": 256
      }
    },
    "vision": {
      "ncnn_models": [
        {
          "name": "YOLOv5s",
          "path": "$MODELS_DIR/vision/yolov5s.param",
          "task": "object_detection"
        },
        {
          "name": "MobileNetV2",
          "path": "$MODELS_DIR/vision/mobilenet_v2.param",
          "task": "classification"
        }
      ]
    },
    "speech": {
      "tts": {
        "voices": [
          {
            "name": "en_US-lessac-medium",
            "path": "$MODELS_DIR/speech/en_US-lessac-medium.onnx",
            "language": "en-US",
            "gender": "male"
          },
          {
            "name": "en_US-amy-medium",
            "path": "$MODELS_DIR/speech/en_US-amy-medium.onnx",
            "language": "en-US",
            "gender": "female"
          }
        ]
      },
      "stt": {
        "primary": {
          "name": "Sherpa-ONNX Streaming Zipformer",
          "path": "$MODELS_DIR/speech/sherpa-onnx-streaming-zipformer-en-2023-06-26",
          "language": "en"
        }
      }
    }
  }
}
EOF
    
    log_success "Model registry created: $REGISTRY_FILE"
}

# Print summary
print_summary() {
    echo ""
    echo "=========================================="
    echo "  Model Download Summary"
    echo "=========================================="
    echo ""
    echo "Models Directory: $MODELS_DIR"
    echo "Model Size: $MODEL_SIZE"
    echo ""
    
    if [ -d "$MODELS_DIR" ]; then
        TOTAL_SIZE=$(du -sh "$MODELS_DIR" | cut -f1)
        echo "Total Size: $TOTAL_SIZE"
        echo ""
        echo "Downloaded Models:"
        find "$MODELS_DIR" -type f \( -name "*.gguf" -o -name "*.onnx" -o -name "*.bin" \) -exec ls -lh {} \; | awk '{print "  - " $9 " (" $5 ")"}'
    fi
    
    echo ""
    echo "Model Registry: $MODELS_DIR/model_registry.json"
    echo ""
    echo "=========================================="
}

# Main function
main() {
    log_info "Starting model download process"
    log_info "Model type: $MODEL_TYPE"
    log_info "Model size: $MODEL_SIZE"
    echo ""
    
    create_model_dirs
    
    case $MODEL_TYPE in
        all)
            download_llm_models
            download_embedding_models
            download_vision_models
            download_speech_models
            download_tokenizers
            ;;
        llm)
            download_llm_models
            download_embedding_models
            download_tokenizers
            ;;
        vision)
            download_vision_models
            ;;
        speech)
            download_speech_models
            ;;
        *)
            log_error "Unknown model type: $MODEL_TYPE"
            exit 1
            ;;
    esac
    
    verify_models
    create_model_registry
    
    echo ""
    log_success "Model download completed successfully!"
    print_summary
}

# Run main
main
