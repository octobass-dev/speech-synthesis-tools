# Advanced TTS Voice Processing System - Complete Guide

### Folder structure
speech-synthesis-vst
    - main.py 
        demo : testing end-to-end tts, voice transfer, speech processing on a single file
    - tts_system.py : End-to-end TTS functions/classes
        - Audio processing
            - Noise reduction   : Noisereduce
            - Pitch             : librosa (pitch correction, midi)
            - Time stretch      : pyrubberbandgain
            - Gain              : pyloudnorm
            - Eq                : scipy signal
            - Reverb            : **TODO** (impulse response based)
        - Speech Synthesis
            - Voice cloning         : SpeechT5
            - emotion modulation    : Add scaled noise to speaker embedding. TODO : trained emotional embeddings
            - Accent                : **TODO**
    - generate_maskgct_audio.py
        Inference script for tts using text, speaker audio, target length(seconds)
    - audio_utils
        Modular functions for self-contained operations
    - **TODO** : Audio Processing into a vst plugin
    - **TODO** transcribe_subs.py : Implement VibeVoice long form transcription, save to .sub file
    ...
    Other files / directory structure
    ...
    - voices 
        Directory with speaker samples, their transcriptions
    - outputs
        Directory (default) to save generated audio
    - test.ipynb
    - requirements.txt
    - models (symlink from Amphion/models)
    - utils (symlink from Amphion/utils)


## 🎯 Overview

This system provides:
- **Voice Cloning**: Clone any voice with 30+ seconds of audio
- **Emotional Speech**: Control emotions and intensity
- **Audio Processing**: Noise removal, pitch correction, time stretching
- **DAW Integration**: Export to any DAW or use as Python library
- **Real-time Control**: VST-style parameter control

## 🚀 Quick Start

### 1. Basic Text-to-Speech

```python
from tts_system import TTSPipeline

# Initialize
pipeline = TTSPipeline()

# Generate speech
pipeline.process_text_to_speech(
    text="Hello, this is a test",
    reference_voice="my_voice.wav",
    output_path="output.wav",
    emotion="happy",
    emotion_intensity=0.8
)
```

### 2. Voice Cloning

```python
from tts_system import AdvancedTTS

tts = AdvancedTTS()
tts.load_voice_cloning_model()

# Clone voice (requires 30+ seconds of clean audio)
audio = tts.clone_voice(
    reference_audio_path="reference_voice.wav",
    text="Any text you want to say",
    emotion="excited",
    emotion_intensity=1.2
)

# Save output
import soundfile as sf
sf.write("cloned_output.wav", audio, 16000)
```

### 3. Audio Processing

```python
from tts_system import AudioProcessor
import soundfile as sf

processor = AudioProcessor(sample_rate=16000)

# Load audio
audio, sr = sf.read("input.wav")

# Remove noise
cleaned = processor.remove_noise(audio)

# Pitch correction
corrected = processor.pitch_correction(cleaned)

# Time stretch (make 20% slower)
stretched = processor.time_stretch(corrected, rate=1.2)

# Normalize loudness
final = processor.normalize_loudness(stretched, target_loudness=-16.0)

# Save
sf.write("processed.wav", final, sr)
```

## 🎛️ VST-Style Processing

### Using the VST Processor

```python
from vst_plugin_python import TTSVSTProcessor

# Initialize processor
processor = TTSVSTProcessor(sample_rate=44100)

# Set parameters (like VST knobs)
processor.set_parameter('emotion', 'happy')
processor.set_parameter('emotion_intensity', 0.8)
processor.set_parameter('pitch_shift', 2.0)  # +2 semitones
processor.set_parameter('reverb_mix', 0.3)
processor.set_parameter('eq_low', 2.0)  # +2dB at 200Hz

# Process text to audio
audio = processor.process_text_to_audio(
    text="Your text here",
    reference_voice="voice.wav",
    output_path="output.wav"
)

# Or process existing audio
import soundfile as sf
input_audio, sr = sf.read("input.wav")
processed = processor.process_audio(input_audio)
```

### Save/Load Presets

```python
# Save current settings
processor.export_preset("my_preset.json")

# Load preset
processor.load_preset("my_preset.json")
```

## 🎨 Advanced Features

### 1. Iterative Segment Editing

Replace specific parts of audio:

```python
from tts_system import AudioProcessor
import soundfile as sf

processor = AudioProcessor()

# Load original audio
original, sr = sf.read("speech.wav")

# Generate new segment
new_segment = tts.clone_voice(
    reference_audio_path="voice.wav",
    text="Corrected phrase"
)

# Replace segment at 5.0 seconds with crossfade
result = processor.replace_segment(
    original_audio=original,
    new_segment=new_segment,
    start_time=5.0,
    crossfade_duration=0.1  # 100ms crossfade
)

sf.write("edited.wav", result, sr)
```

### 2. Pitch Correction to Musical Notes

```python
# Define chord sequence (MIDI notes)
chord_sequence = [60, 64, 67]  # C major triad (C, E, G)

# Correct pitch to match chord
corrected = processor.pitch_correction(
    audio,
    target_notes=chord_sequence,
    correction_strength=0.8  # 80% correction
)
```

### 3. Multiple Emotion Blending

```python
# Generate same text with different emotions
emotions = ['neutral', 'happy', 'sad']
outputs = []

for emotion in emotions:
    audio = tts.clone_voice(
        "voice.wav",
        "This is a test",
        emotion=emotion,
        emotion_intensity=0.7
    )
    outputs.append(audio)

# Blend emotions (custom weighted mix)
blended = 0.5 * outputs[0] + 0.3 * outputs[1] + 0.2 * outputs[2]
```

### 4. Batch Processing

```python
texts = [
    "First sentence.",
    "Second sentence.",
    "Third sentence."
]

for i, text in enumerate(texts):
    pipeline.process_text_to_speech(
        text=text,
        reference_voice="voice.wav",
        output_path=f"output_{i}.wav",
        emotion="neutral"
    )
```

## 🎹 DAW Integration

### Option 1: DawDreamer Graph

```python
import dawdreamer as daw

# Create DAW engine
engine = daw.RenderEngine(44100, 512)

# Generate TTS
processor = TTSVSTProcessor()
audio = processor.process_text_to_audio(
    "Your text",
    "voice.wav"
)

# Create playback node
playback = engine.make_playback_processor("tts", audio)

# Load VST effects
reverb = engine.make_plugin_processor(
    "reverb", 
    "/path/to/reverb.vst3"
)

# Create processing graph
engine.load_graph([
    (playback, []),
    (reverb, [playback.get_name()])
])

# Render
engine.render(duration_seconds=10.0)

# Get output
output = reverb.get_audio()
```

### Option 2: Export Audio Files

```python
# Process and export
processor.process_text_to_audio(
    text="Your narration",
    reference_voice="voice.wav",
    output_path="narration.wav"
)

# Import into DAW:
# - Drag narration.wav into your DAW
# - Apply additional processing
# - Mix with music/other tracks
```

### Option 3: REAPER Integration

Create a REAPER JSFX script:

```javascript
desc: TTS Voice Processor
slider1:0<0,1,0.01>Emotion Intensity
slider2:0<-12,12,0.1>Pitch Shift

@init
// Initialize

@slider
emotion_intensity = slider1;
pitch_shift = slider2;

@sample
// Process each sample
spl0 = spl0;  // Left
spl1 = spl1;  // Right
```

Save as `TTS_Processor.jsfx` in REAPER's Effects folder.

## 🎤 Preparing Reference Audio

For best voice cloning results:

### Requirements
- **Duration**: 30-60 seconds minimum
- **Quality**: Clean, minimal background noise
- **Content**: Natural speech, varied intonation
- **Format**: WAV, 16-44.1 kHz sample rate

### Tips
```python
# Pre-process reference audio
from tts_system import AudioProcessor

processor = AudioProcessor()

# Load reference
ref_audio, sr = sf.read("raw_reference.wav")

# Clean up
cleaned = processor.remove_noise(ref_audio)
normalized = processor.normalize_loudness(cleaned, -20.0)

# Save processed reference
sf.write("clean_reference.wav", normalized, sr)
```

## 🔧 Parameter Reference

### Emotion Parameters
- **emotion**: `"neutral"`, `"happy"`, `"sad"`, `"angry"`, `"excited"`, `"calm"`
- **emotion_intensity**: `0.0` (none) to `2.0` (extreme), default `0.5`

### Audio Processing
- **pitch_shift**: `-12.0` to `+12.0` semitones
- **time_stretch**: `0.5` (2x faster) to `2.0` (2x slower)
- **noise_reduction**: `0.0` (none) to `1.0` (maximum)
- **loudness_target**: `-30.0` to `0.0` LUFS (standard: -16.0)

### Effects
- **reverb_mix**: `0.0` (dry) to `1.0` (wet)
- **eq_low**: `-12.0` to `+12.0` dB at 200 Hz
- **eq_mid**: `-12.0` to `+12.0` dB at 1 kHz
- **eq_high**: `-12.0` to `+12.0` dB at 5 kHz

## 📊 Performance Optimization

### GPU Acceleration

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("Using CPU")

# Initialize with GPU
tts = AdvancedTTS(device=device)
```

### Batch Processing

```python
# Process multiple texts efficiently
texts = ["Text 1", "Text 2", "Text 3"]
speaker_embedding = tts.extract_speaker_embedding("voice.wav")

# Reuse embedding
for text in texts:
    audio = tts.synthesize_speech(text, speaker_embedding)
    # Process audio...
```

### Model Caching

```python
# Pre-load models
tts.load_voice_cloning_model()
_ = tts.extract_speaker_embedding("voice.wav")

# Now synthesis is faster
# (models are cached in memory)
```

## 🎯 Use Cases

### 1. Audiobook Narration

```python
# Split book into chapters
chapters = [...]

for i, chapter in enumerate(chapters):
    pipeline.process_text_to_speech(
        text=chapter,
        reference_voice="narrator_voice.wav",
        output_path=f"chapter_{i}.wav",
        emotion="neutral",
        target_loudness=-20.0  # Audiobook standard
    )
```

### 2. Podcast Production

```python
# Generate introduction
intro = pipeline.process_text_to_speech(
    "Welcome to the podcast!",
    reference_voice="host.wav",
    emotion="excited",
    emotion_intensity=0.9
)

# Add processing
processor = AudioProcessor()
intro_processed = processor.normalize_loudness(intro, -16.0)
```

### 3. Video Game Dialogue

```python
# Generate NPC dialogue with emotions
dialogues = {
    "greeting": ("Hello, traveler!", "happy"),
    "warning": ("Danger ahead!", "concerned"),
    "farewell": ("Safe travels!", "calm")
}

for line_id, (text, emotion) in dialogues.items():
    audio = tts.clone_voice(
        "npc_voice.wav",
        text,
        emotion=emotion
    )
    sf.write(f"npc_{line_id}.wav", audio, 16000)
```

### 4. Accessibility Tools

```python
# Screen reader with emotional context
def speak_with_emotion(text, context="neutral"):
    emotion_map = {
        "error": "concerned",
        "success": "happy",
        "warning": "neutral",
        "info": "calm"
    }
    
    audio = tts.clone_voice(
        "reader_voice.wav",
        text,
        emotion=emotion_map.get(context, "neutral")
    )
    
    # Play audio in real-time
    # (requires pyaudio or similar)
```

## 🐛 Troubleshooting

### Poor Voice Clone Quality
- Use 60+ seconds of reference audio
- Ensure clean audio (remove noise first)
- Increase emotion_intensity for more expression
- Try different text samples

### Robotic Sound
- Reduce pitch correction strength
- Lower noise reduction amount
- Adjust emotion_intensity
- Check sample rate matches

### Slow Processing
- Enable GPU acceleration
- Reduce audio length
- Cache speaker embeddings
- Use smaller models

### Memory Issues
- Process in smaller batches
- Clear GPU cache: `torch.cuda.empty_cache()`
- Use CPU for very long audio
- Reduce buffer size

## 📚 Additional Resources

- [SpeechBrain Documentation](https://speechbrain.github.io/)
- [Transformers TTS Guide](https://huggingface.co/docs/transformers/tasks/text-to-speech)
- [DawDreamer Examples](https://github.com/DBraun/DawDreamer/tree/main/examples)
- [JUCE Tutorials](https://docs.juce.com/master/tutorial_create_projucer_basic_plugin.html)
- [Audio DSP Resources](https://github.com/BillyDM/Awesome-Audio-DSP)

## 🤝 Contributing

Improve this system:
1. Add new emotion presets
2. Integrate additional TTS models
3. Create more DSP effects
4. Build DAW-specific integrations
5. Optimize performance

## 📄 License

This code is provided as-is for educational and commercial use. Please respect voice cloning ethics and obtain consent before cloning someone's voice.