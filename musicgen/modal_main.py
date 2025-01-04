import modal
from io import BytesIO
from fastapi import FastAPI, HTTPException
import base64
from typing import Dict, Tuple, Any

# Create a persistent volume to store model files across runs
# This prevents having to re-download models every time
volume = modal.Volume.from_name("models")
VOLUME_PATH = "/root/models"

# Initialize FastAPI and Modal apps
app = modal.App("musicgen-api")
web_app = FastAPI()

# Modal Dict is a persistent key-value store that persists across function calls
model_cache = modal.Dict.from_name("musicgen-model-cache", create_if_missing=True)

# In-memory cache to store loaded models during the container's lifetime
# This helps avoid reloading models between generations while the container is alive
_memory_cache: Dict[str, Tuple[Any, Any]] = {}

# Define the container image with all required dependencies
# This image will be used to run our functions in the cloud
musicgen_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy",
        "scipy",
        "transformers",
        "accelerate>=0.26.0",
    )
    # Install PyTorch with CUDA 11.8 support for GPU acceleration
    .pip_install(
        "torch",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu118"
    )
    # Set Hugging Face cache directory to our persistent volume
    .env({"HF_HOME": f"{VOLUME_PATH}/hf_cache"})
)

# Function to download model files to persistent storage
@app.function(
    image=musicgen_image,
    volumes={VOLUME_PATH: volume},
    cpu=1.0,  # Only need CPU for downloads
)
def download_model(model_name: str = "facebook/musicgen-medium"):
    """Download the model files to the persistent volume."""
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    import os
    
    model_path = f"{VOLUME_PATH}/{model_name.split('/')[-1]}"
    print(f"Downloading model to {model_path}...")
    
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        
        os.makedirs(model_path, exist_ok=True)
        processor.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        return {"status": "success", "message": f"Model {model_name} downloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")

# Internal helper function to load and cache models
def _internal_get_model_and_processor(model_name: str = "facebook/musicgen-medium"):
    """Internal function to load and cache the model and processor.
    Uses a two-level caching strategy:
    1. In-memory cache for fastest access
    2. Persistent volume storage for when container restarts
    """
    import os
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    # Check in-memory cache first (fastest)
    if model_name in _memory_cache:
        print("Using in-memory cached model and processor...")
        processor, state_dict = _memory_cache[model_name]
        # Even with cached state_dict, we need to initialize model architecture
        # We use float16 (half precision) to reduce memory usage
        model = MusicgenForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=f"{VOLUME_PATH}/{model_name.split('/')[-1]}",
            state_dict=state_dict,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda()  # Move model to GPU
        return processor, model

    # Check if model files exist in volume
    model_path = f"{VOLUME_PATH}/{model_name.split('/')[-1]}"
    if not os.path.exists(model_path):
        print("Model not found in volume, downloading first...")
        download_model.remote(model_name)

    print("Loading model and processor into memory...")
    try:
        # Load processor
        processor = AutoProcessor.from_pretrained(model_path)

        # Load model and its state_dict
        model = MusicgenForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        state_dict = model.state_dict()

        # Cache processor and state_dict in memory
        _memory_cache[model_name] = (processor, state_dict)
        print("Model and processor cached in memory!")

        # Move the model to GPU
        model = model.cuda()
        return processor, model

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.function(
    image=musicgen_image,
    volumes={VOLUME_PATH: volume},
    cpu=1.0,
    timeout=1000,
)
def get_model_and_processor(model_name: str = "facebook/musicgen-medium"):
    """Modal function wrapper for loading and caching the model."""
    return _internal_get_model_and_processor(model_name)

# Main music generation function that runs on GPU
@app.function(
    image=musicgen_image,
    volumes={VOLUME_PATH: volume},
    gpu="A100-80GB",  # Request an A100 GPU with 80GB memory
    timeout=1000,     # Maximum runtime in seconds
    container_idle_timeout=60,  # Keep container alive for 60s after completion
)
def generate_music(prompt: str, duration: int = 5, model_name: str = "facebook/musicgen-medium") -> bytes:
    import scipy.io.wavfile
    from torch.cuda.amp import autocast  # For automatic mixed precision (faster/less memory)
    import time
    import torch

    print(f"Starting music generation for prompt: '{prompt}' ({duration} seconds)")
    
    processor, model = _internal_get_model_and_processor(model_name)
    
    # MusicGen generates ~50 tokens per second of audio
    max_new_tokens = int(duration * 50)

    # Convert text prompt to model inputs
    print("Processing input prompt...")
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to("cuda")  # Move inputs to GPU

    # Generate the audio using automatic mixed precision for efficiency
    print(f"Generating audio with {max_new_tokens} tokens...")
    start_time = time.time()
    torch.cuda.synchronize()  # Ensure GPU operations are complete
    with autocast(enabled=True):
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Audio generation completed in {end_time - start_time:.2f} seconds!")

    # Convert the generated audio to WAV format
    print("Converting to WAV format...")
    sampling_rate = model.config.audio_encoder.sampling_rate
    buffer = BytesIO()
    audio_data = audio_values[0, 0].cpu().to(torch.float32).numpy()
    scipy.io.wavfile.write(buffer, rate=sampling_rate, data=audio_data)
    print("Conversion to WAV complete!")
    
    return buffer.getvalue()

# API endpoints below handle the web interface to our functions

# Endpoint to trigger model downloads
@web_app.post("/download-model")
async def download_model_endpoint(model_name: str = "facebook/musicgen-medium"):
    print(f"Received request to download model: {model_name}")
    function_call = download_model.spawn(model_name)
    try:
        result = function_call.get(timeout=600)  # 10 minute timeout for downloads
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Endpoint to start music generation
# Returns a task ID that can be used to check generation status
@web_app.get("/generate")
async def generate_music_endpoint(prompt: str, duration: int = 5, model_name: str = "facebook/musicgen-medium"):
    print(f"Received request for prompt: '{prompt}' ({duration} seconds)")
    function_call = generate_music.spawn(prompt, duration, model_name)
    return {"task_id": function_call.object_id}

# Endpoint to check the status of a generation task
# Returns base64 encoded audio when complete
@web_app.get("/status/{task_id}")
async def check_status(task_id: str):
    try:
        call = modal.functions.FunctionCall.from_id(task_id)
        try:
            output = call.get(timeout=0)
            return {
                "status": "completed",
                "result": base64.b64encode(output).decode('utf-8')
            }
        except TimeoutError:
            return {
                "status": "running"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Mount the FastAPI app as a Modal endpoint
@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app