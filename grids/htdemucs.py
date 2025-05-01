import librosa
import torchaudio
from models.htdemucs import HTDemucsModel
import torch

def print_progress(current, total):
    print(f"{current} / {total}")

def process_audio(input_file="test.mp3"):
    model = HTDemucsModel(["drums", "bass", "other", "vocals"],"weights/htdemucs/75fc33f5-1941ce65.th", model_included_in_path=True, segment_callback=print_progress, device="cuda")
    waveform, sr = librosa.load(input_file, sr=44100, mono=False)

    sources = model.separate(waveform)
    for source in ["drums", "bass", "other", "vocals"]:
        torchaudio.save(f"output_{source}.wav", torch.tensor(sources[source]).cpu() , 44100)

with torch.no_grad():
    process_audio()