import whisper
import os
from tqdm import tqdm
import torch

# Directory containing the audio files
audio_directory = "/Users/laurikeskull/Documents/Programming/meetings/src/audios"
# Directory to save transcriptions
output_directory = "/Users/laurikeskull/Documents/Programming/meetings/src/transcriptions"

def transcribe_audio(model, audio_path):
    """Transcribe a single audio file."""
    result = model.transcribe(audio_path)
    return result["text"]

def main():

    device = torch.device("cpu")
    print("Apple Metal not available, using CPU")

    # Load the Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("turbo", device=device)  # Using the latest "turbo" model

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get all .m4a files in the audio directory
    audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.m4a')]

    print(f"Found {len(audio_files)} .m4a files to transcribe.")

    # Process each audio file
    for audio_file in tqdm(audio_files, desc="Transcribing"):
        audio_path = os.path.join(audio_directory, audio_file)
        output_path = os.path.join(output_directory, f"{os.path.splitext(audio_file)[0]}.txt")

        # Transcribe the audio
        transcription = transcribe_audio(model, audio_path)

        # Save the transcription
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription)

    print("Transcription complete. Results saved in the transcriptions directory.")

if __name__ == "__main__":
    main()
