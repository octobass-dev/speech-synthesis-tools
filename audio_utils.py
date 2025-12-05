from faster_whisper import WhisperModel

def transcribe_audio(audio_file, model = None, modelname="large-v3"):
    """
        audio_file  : file to transcribe
        model       : Already loaded model in case of multiple jobs
        modelname   : In case no model is provided, load this model
    """
    if model is None:
        model = WhisperModel(modelname, device="cuda", compute_type="float16")
    segments, info = model.transcribe(audio_file, beam_size=5)
    # TODO : Process segments to mark newline.
    text = ' '.join([x.text for x in segments])
    return text, segments, info