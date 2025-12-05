"""
Advanced Text-to-Speech and Voice Processing System
Supports voice cloning, emotional synthesis, noise removal, pitch correction
"""

import torch
import torchaudio
from transformers import AutoTokenizer, AutoModel
import soundfile as sf
import numpy as np
from typing import Optional, Dict, List, Tuple
import librosa
import noisereduce as nr
from scipy import signal
import pyloudnorm as pyln
import pyrubberband as pyrb

class AdvancedTTS:
    def __init__(self, device: str = None):
        """Initialize TTS system with various models"""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 16000
        self.models = {}
        
    def load_voice_cloning_model(self, model_name: str = "microsoft/speecht5_tts"):
        """Load voice cloning model"""
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        
        print(f"Loading voice cloning model: {model_name}")
        self.models['processor'] = SpeechT5Processor.from_pretrained(model_name)
        self.models['tts'] = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(self.device)
        self.models['vocoder'] = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
    
    def extract_speaker_embedding(self, audio_path: str) -> torch.Tensor:
        """Extract speaker embedding from reference audio"""
        from speechbrain.pretrained import EncoderClassifier
        
        if 'speaker_model' not in self.models:
            self.models['speaker_model'] = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir="pretrained_models/spkrec"
            )
        
        # Load and process audio
        signal, fs = torchaudio.load(audio_path)#, backend='soundfile')
        
        # Resample if necessary
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
        
        # Convert stereo to mono
        if signal.ndim > 1:
            signal = torch.mean(signal, axis=0)
        
        signal = signal.reshape([1,-1])
        #print(signal.shape, signal.ndim, signal.dtype)
        # Extract embedding
        with torch.no_grad():
            embedding = self.models['speaker_model'].encode_batch(signal)
        
        return embedding.squeeze()
    
    def synthesize_speech(
        self, 
        text: str, 
        speaker_embedding: torch.Tensor,
        emotion: str = "neutral",
        emotion_intensity: float = 1.0
    ) -> np.ndarray:
        """
        Synthesize speech with voice cloning and emotion control
        
        Args:
            text: Input text to synthesize
            speaker_embedding: Speaker embedding from reference audio
            emotion: Emotion type (neutral, happy, sad, angry, excited, calm)
            emotion_intensity: Intensity of emotion (0.0 to 2.0)
        """
        if 'processor' not in self.models:
            self.load_voice_cloning_model()
        
        # Process text
        inputs = self.models['processor'](text=text, return_tensors="pt")
        # Apply emotion modulation to speaker embedding
        modulated_embedding = self._apply_emotion_modulation(
            speaker_embedding, 
            emotion, 
            emotion_intensity
        )
        
        modulated_embedding = modulated_embedding.reshape([1,-1])
        # print(speaker_embedding.shape, modulated_embedding.shape)

        # Generate speech
        with torch.no_grad():
            speech = self.models['tts'].generate_speech(
                inputs["input_ids"].to(self.device),
                modulated_embedding.to(self.device),
                vocoder=self.models['vocoder']
            )
        
        return speech.cpu().numpy()
    
    def _apply_emotion_modulation(
        self, 
        embedding: torch.Tensor, 
        emotion: str, 
        intensity: float
    ) -> torch.Tensor:
        """
        Apply emotion-specific modulation to speaker embedding
        This is a simplified version - production systems use trained emotion vectors
        """
        emotion_vectors = {
            'neutral': torch.zeros_like(embedding),
            'happy': torch.randn_like(embedding) * 0.1,
            'sad': torch.randn_like(embedding) * -0.1,
            'angry': torch.randn_like(embedding) * 0.15,
            'excited': torch.randn_like(embedding) * 0.2,
            'calm': torch.randn_like(embedding) * -0.05,
            'warm': torch.randn_like(embedding) * 0.05,
        }
        
        emotion_vector = emotion_vectors.get(emotion, emotion_vectors['neutral'])
        modulated = embedding + emotion_vector * intensity
        
        return modulated
    
    def clone_voice(
        self, 
        reference_audio_path: str, 
        text: str,
        emotion: str = "neutral",
        emotion_intensity: float = 1.0
    ) -> np.ndarray:
        """
        Clone voice from reference audio and synthesize new text
        
        Args:
            reference_audio_path: Path to reference audio (30s+ recommended)
            text: Text to synthesize in cloned voice
            emotion: Desired emotion
            emotion_intensity: Intensity of emotion
        """
        # Extract speaker characteristics
        speaker_embedding = self.extract_speaker_embedding(reference_audio_path)
        
        # Synthesize with cloned voice
        audio = self.synthesize_speech(text, speaker_embedding, emotion, emotion_intensity)
        
        return audio


class AudioProcessor:
    def __init__(self, sample_rate: int = 16000):
        """Initialize audio processing tools"""
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(sample_rate)
    
    def remove_noise(
        self, 
        audio: np.ndarray, 
        noise_profile: Optional[np.ndarray] = None,
        stationary: bool = True
    ) -> np.ndarray:
        """
        Remove background noise from audio
        
        Args:
            audio: Input audio array
            noise_profile: Optional noise profile for reduction
            stationary: Whether noise is stationary
        """
        # Use noisereduce library
        if noise_profile is not None:
            reduced = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                y_noise=noise_profile,
                stationary=stationary
            )
        else:
            # Auto-detect noise
            reduced = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=stationary
            )
        
        return reduced
    
    def pitch_correction(
        self, 
        audio: np.ndarray,
        target_notes: Optional[List[float]] = None,
        correction_strength: float = 0.5
    ) -> np.ndarray:
        """
        Apply pitch correction to audio
        
        Args:
            audio: Input audio
            target_notes: Target MIDI notes for correction
            correction_strength: How aggressively to correct (0.0 to 1.0)
        """
        # Extract pitch using librosa
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        # Quantize to nearest semitone if no target notes specified
        if target_notes is None:
            # Auto-tune to nearest semitone
            f0_corrected = np.copy(f0)
            valid_f0 = f0[~np.isnan(f0)]
            
            if len(valid_f0) > 0:
                midi_notes = librosa.hz_to_midi(valid_f0)
                corrected_midi = np.round(midi_notes)
                f0_corrected[~np.isnan(f0)] = librosa.midi_to_hz(
                    midi_notes + (corrected_midi - midi_notes) * correction_strength
                )
        else:
            # Correct to specific target notes
            f0_corrected = self._correct_to_target_notes(f0, target_notes, correction_strength)
        
        # Apply pitch shift
        pitch_shift_semitones = 12 * np.log2(
            np.nanmean(f0_corrected[~np.isnan(f0_corrected)]) / 
            np.nanmean(f0[~np.isnan(f0)])
        )
        
        corrected_audio = pyrb.pitch_shift(
            audio,
            self.sample_rate,
            pitch_shift_semitones
        )
        
        return corrected_audio
    
    def _correct_to_target_notes(
        self, 
        f0: np.ndarray, 
        target_notes: List[float],
        strength: float
    ) -> np.ndarray:
        """Correct pitches to target MIDI notes"""
        f0_corrected = np.copy(f0)
        
        for i, freq in enumerate(f0):
            if not np.isnan(freq):
                midi = librosa.hz_to_midi(freq)
                
                # Find nearest target note
                nearest_target = min(target_notes, key=lambda x: abs(x - midi))
                
                # Apply correction
                corrected_midi = midi + (nearest_target - midi) * strength
                f0_corrected[i] = librosa.midi_to_hz(corrected_midi)
        
        return f0_corrected
    
    def time_stretch(
        self, 
        audio: np.ndarray, 
        rate: float
    ) -> np.ndarray:
        """
        Stretch or compress audio without changing pitch
        
        Args:
            audio: Input audio
            rate: Stretch rate (>1.0 = slower, <1.0 = faster)
        """
        stretched = pyrb.time_stretch(audio, self.sample_rate, rate)
        return stretched
    
    def normalize_loudness(
        self, 
        audio: np.ndarray, 
        target_loudness: float = -16.0
    ) -> np.ndarray:
        """
        Normalize audio to target loudness in LUFS
        
        Args:
            audio: Input audio
            target_loudness: Target loudness in LUFS
        """
        # Measure current loudness
        loudness = self.meter.integrated_loudness(audio)
        
        # Calculate gain
        gain = target_loudness - loudness
        
        # Apply gain
        normalized = pyln.normalize.loudness(audio, loudness, target_loudness)
        
        return normalized
    
    def apply_eq(
        self, 
        audio: np.ndarray,
        eq_params: Dict[str, float]
    ) -> np.ndarray:
        """
        Apply parametric EQ
        
        Args:
            audio: Input audio
            eq_params: Dict with 'freq', 'gain', 'q' for each band
        """
        filtered = audio.copy()
        
        for band, params in eq_params.items():
            freq = params['freq']
            gain = params['gain']
            q = params.get('q', 1.0)
            
            # Design filter
            b, a = signal.iirpeak(freq, q, self.sample_rate)
            
            # Apply filter with gain
            filtered = signal.filtfilt(b, a, filtered) * (10 ** (gain / 20))
        
        return filtered
    
    def segment_audio(
        self, 
        audio: np.ndarray,
        segment_start: float,
        segment_end: float
    ) -> np.ndarray:
        """
        Extract a segment from audio
        
        Args:
            audio: Input audio
            segment_start: Start time in seconds
            segment_end: End time in seconds
        """
        start_sample = int(segment_start * self.sample_rate)
        end_sample = int(segment_end * self.sample_rate)
        
        return audio[start_sample:end_sample]
    
    def replace_segment(
        self,
        original_audio: np.ndarray,
        new_segment: np.ndarray,
        start_time: float,
        crossfade_duration: float = 0.05
    ) -> np.ndarray:
        """
        Replace a segment in audio with crossfading
        
        Args:
            original_audio: Original audio
            new_segment: Replacement segment
            start_time: Where to insert replacement
            crossfade_duration: Crossfade length in seconds
        """
        start_sample = int(start_time * self.sample_rate)
        end_sample = start_sample + len(new_segment)
        
        # Create output array
        result = original_audio.copy()
        
        # Calculate crossfade samples
        fade_samples = int(crossfade_duration * self.sample_rate)
        
        # Apply crossfade at start
        if start_sample > fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            new_segment[:fade_samples] = (
                new_segment[:fade_samples] * fade_in +
                result[start_sample:start_sample+fade_samples] * fade_out
            )
        
        # Apply crossfade at end
        if end_sample < len(result) - fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            new_segment[-fade_samples:] = (
                new_segment[-fade_samples:] * fade_out +
                result[end_sample-fade_samples:end_sample] * fade_in
            )
        
        # Insert new segment
        result[start_sample:end_sample] = new_segment
        
        return result


class TTSPipeline:
    def __init__(self):
        """Complete TTS processing pipeline"""
        self.tts = AdvancedTTS()
        self.processor = AudioProcessor()
    
    def process_text_to_speech(
        self,
        text: str,
        reference_voice: str,
        output_path: str,
        emotion: str = "neutral",
        emotion_intensity: float = 1.0,
        remove_noise: bool = True,
        apply_pitch_correction: bool = False,
        target_duration: Optional[float] = None,
        target_loudness: float = -16.0
    ):
        """
        Complete text-to-speech processing pipeline
        
        Args:
            text: Text to synthesize
            reference_voice: Path to reference voice audio
            output_path: Where to save output
            emotion: Emotion to apply
            emotion_intensity: Intensity of emotion
            remove_noise: Whether to apply noise reduction
            apply_pitch_correction: Whether to apply pitch correction
            target_duration: Target duration in seconds (for time stretching)
            target_loudness: Target loudness in LUFS
        """
        # Generate speech
        print("Generating speech...")
        audio = self.tts.clone_voice(
            reference_voice,
            text,
            emotion=emotion,
            emotion_intensity=emotion_intensity
        )
        
        # Post-processing
        if remove_noise:
            print("Removing noise...")
            audio = self.processor.remove_noise(audio)
        
        if apply_pitch_correction:
            print("Applying pitch correction...")
            audio = self.processor.pitch_correction(audio)
        
        if target_duration:
            current_duration = len(audio) / self.processor.sample_rate
            rate = current_duration / target_duration
            print(f"Time stretching (rate: {rate:.2f})...")
            audio = self.processor.time_stretch(audio, rate)
        
        # Normalize loudness
        print("Normalizing loudness...")
        audio = self.processor.normalize_loudness(audio, target_loudness)
        
        # Save
        sf.write(output_path, audio, self.processor.sample_rate)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    pipeline = TTSPipeline()
    
    # Example: Clone voice and generate emotional speech
    pipeline.process_text_to_speech(
        text="Hello! This is a test of the advanced text-to-speech system.",
        reference_voice="path/to/your/voice.wav",
        output_path="output.wav",
        emotion="happy",
        emotion_intensity=0.8,
        remove_noise=True,
        apply_pitch_correction=False,
        target_loudness=-16.0
    )