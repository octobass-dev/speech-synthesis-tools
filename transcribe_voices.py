from pathlib import Path
from faster_whisper import WhisperModel
from audio_utils import transcribe_audio

formats = [".m4a",".mp3", ".mp4", ".mpeg", ".mpga", ".ogg", ".wav", ".webm", ".flac", ".opus"] 
voices_dir = "voices"
if __name__=="__main__":
    p = Path(voices_dir)
    to_generate = []
    files = list(p.iterdir())
    for x in files:
        if x.suffix in formats and not Path(f"{voices_dir}/{x.stem}.txt").is_file():
            to_generate.append(x.resolve())
    if len(to_generate) > 0:
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        for y in to_generate:
            print(y)
            text, segments, info = transcribe_audio(y, model)

            # Print detected language and transcription
            print(f"Detected language '{info.language}' with probability {info.language_probability}")
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            with open(f"{voices_dir}/{y.stem}.txt", "w") as f:
                f.write(text)
                
    