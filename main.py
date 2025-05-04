import torchaudio
from kaumena.models import HTDemucsModel, OpenUnmixModel
import torch
import librosa

def print_progress(current, total):
    print(f"{current} / {total}")


def process_audio(input_file="Amourski - Я не спал (Official Track).mp3"):
    #model = OpenUnmixModel(["drums", "bass", "other", "vocals"], model_type="umxhq", device="cuda", segment_callback=print_progress)
    model = HTDemucsModel(["drums", "bass", "other", "vocals"],"weights/htdemucs/75fc33f5-1941ce65.th", model_included_in_path=True, segment_callback=print_progress, device="cuda")
    waveform, sr = librosa.load(input_file, sr=44100, mono=False)

    sources = model.separate(waveform)
    for source in ["drums", "bass", "other", "vocals"]:
        output = sources[source]
        torchaudio.save(f"output_{source}.mp3", torch.tensor(output).cpu() , 44100)

with torch.no_grad():
    process_audio()



