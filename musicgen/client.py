import requests
import argparse
import time
import base64
import os

def download_model(base_url: str, model_name: str = "facebook/musicgen-medium"):
    """Download a model to the Modal volume"""
    print(f"Requesting download of model: {model_name}")
    response = requests.post(f"{base_url}/download-model", params={"model_name": model_name})
    
    if response.status_code != 200:
        print(f"Error downloading model: {response.status_code}")
        print(response.text)
        return False
    
    result = response.json()
    if result.get("status") == "success":
        print(f"Model downloaded successfully: {result.get('message')}")
        return True
    else:
        print(f"Error: {result.get('message')}")
        return False

def generate_music(prompt: str, duration: int = 5, output_file: str = "output.wav", model_name: str = "facebook/musicgen-medium"):
    """
    Call the Modal API endpoint to generate music and save it to a file.
    """
    base_url = os.getenv("MODAL_APP_URL")
    
    # Start the generation
    print(f"Requesting music generation for prompt: '{prompt}' ({duration} seconds)")
    response = requests.get(f"{base_url}/generate", params={
        "prompt": prompt,
        "duration": duration,
        "model_name": model_name
    })
    
    if response.status_code == 404 and "Model not found" in response.text:
        print("Model not found. Attempting to download...")
        if not download_model(base_url, model_name):
            print("Failed to download model. Aborting.")
            return
        # Retry generation after download
        response = requests.get(f"{base_url}/generate", params={
            "prompt": prompt,
            "duration": duration,
            "model_name": model_name
        })
    
    if response.status_code != 200:
        print(f"Error starting generation: {response.status_code}")
        print(response.text)
        return

    task_id = response.json()["task_id"]
    print(f"Generation started! Task ID: {task_id}")
    
    # Poll for completion
    while True:
        print("Checking status...")
        status_response = requests.get(f"{base_url}/status/{task_id}")
        
        if status_response.status_code != 200:
            print(f"Error checking status: {status_response.status_code}")
            print(status_response.text)
            return
        
        status_data = status_response.json()
        
        if status_data["status"] == "completed":
            print("Generation complete! Saving file...")
            audio_data = base64.b64decode(status_data["result"])
            with open(output_file, "wb") as f:
                f.write(audio_data)
            print(f"Successfully saved audio to {output_file}")
            break
        elif status_data["status"] == "error":
            print(f"Error in generation: {status_data.get('error', 'Unknown error')}")
            break
        else:
            print("Still generating... checking again in 10 seconds")
            time.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate music using Modal API')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Download model command
    download_parser = subparsers.add_parser('download', help='Download a model')
    download_parser.add_argument('--model', '-m', type=str, default="facebook/musicgen-medium",
                               help='Model to download (default: facebook/musicgen-medium)')
    
    # Generate music command
    generate_parser = subparsers.add_parser('generate', help='Generate music')
    generate_parser.add_argument('prompt', type=str, help='Text description of the music to generate')
    generate_parser.add_argument('--duration', '-d', type=int, default=5,
                               help='Duration in seconds (default: 5)')
    generate_parser.add_argument('--output', '-o', type=str, default="output.wav",
                               help='Output file path (default: output.wav)')
    generate_parser.add_argument('--model', '-m', type=str, default="facebook/musicgen-medium",
                               help='Model to use (default: facebook/musicgen-medium)')
    
    args = parser.parse_args()
    base_url = os.getenv("MODAL_APP_URL")

    # If duration exceeds 30 seconds, notify the user and bail out
    if args.duration > 30:
        print("MusicGen only supports a max of 30 second generations. Please try again with a shorter duration.")
        exit(1)

    if args.command == 'download':
        download_model(base_url, args.model)
    elif args.command == 'generate':
        generate_music(args.prompt, args.duration, args.output, args.model)
    else:
        parser.print_help() 