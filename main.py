from tts_system import TTSPipeline
if __name__=="__main__":
    # Initialize
    pipeline = TTSPipeline()

    # Generate speech
    pipeline.process_text_to_speech(
        text="Relax your whole body, close your eyes, and make yourself calm and quiet.",
        reference_voice="voices/gettysburg_address.mp3",
        output_path="outputs/ttsp_verse1_1.wav",
        emotion="happy",
        emotion_intensity=0.2
    )
