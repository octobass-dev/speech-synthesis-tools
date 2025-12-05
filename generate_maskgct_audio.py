# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from huggingface_hub import hf_hub_download
import safetensors
import soundfile as sf
import argparse
import os

from audio_utils import transcribe_audio

# Amphion packages
import sys
amphion_home = os.environ['AMPHION_HOME']
sys.path.append(amphion_home)

from models.tts.maskgct.maskgct_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--guide-audio', default='./voices/timothy_voice.mp3', help="Speech sample for guidance")
    parser.add_argument('-o', '--out-audio', default='./outputs/generated.mp3', help="Output file name")
    parser.add_argument('-t', '--target-text', default='Awareness, Impermanence, Perseverance.', help="Text to generate")
    parser.add_argument('-l', '--target-len', default=None, type=int, help="Expected duration of audio in seconds.")
    args = parser.parse_args()

    # build model
    device = torch.device("cuda:0")
    cfg_path = f"{amphion_home}/models/tts/maskgct/config/maskgct.json"
    cfg = load_config(cfg_path)
    # 1. build semantic model (w2v-bert-2.0)
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    # 2. build semantic codec
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    # 3. build acoustic codec
    codec_encoder, codec_decoder = build_acoustic_codec(
        cfg.model.acoustic_codec, device
    )
    # 4. build t2s model
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    # 5. build s2a model
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

    # download checkpoint
    # download semantic codec ckpt
    semantic_code_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
    )
    # download acoustic codec ckpt
    codec_encoder_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="acoustic_codec/model.safetensors"
    )
    codec_decoder_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="acoustic_codec/model_1.safetensors"
    )
    # download t2s model ckpt
    t2s_model_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="t2s_model/model.safetensors"
    )
    # download s2a model ckpt
    s2a_1layer_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="s2a_model/s2a_model_1layer/model.safetensors"
    )
    s2a_full_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="s2a_model/s2a_model_full/model.safetensors"
    )

    # load semantic codec
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    # load acoustic codec
    safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
    safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
    # load t2s model
    safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
    # load s2a model
    safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
    safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

    # inference
    prompt_wav_path = args.guide_audio
    text_path = f"{prompt_wav_path.rsplit('.', 1)[0]}.txt"
    print(text_path)
    if os.path.exists(text_path):
        with open(text_path, 'r') as txtf:
            prompt_text = txtf.read()
    else:
        prompt_text = transcribe_audio(prompt_wav_path)[0]
    print(prompt_text)
    #prompt_text = " We do not break. We never give in. We never back down."
    #target_text = "In this paper, we introduce MaskGCT, a fully non-autoregressive TTS model that eliminates the need for explicit alignment information between text and speech supervision."
    # Specify the target duration (in seconds). If target_len = None, we use a simple rule to predict the target duration.
    
    # target_len = 18
    maskgct_inference_pipeline = MaskGCT_Inference_Pipeline(
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
        device,
    )

    recovered_audio = maskgct_inference_pipeline.maskgct_inference(
        args.guide_audio, prompt_text, args.target_text, "en", "en", target_len=args.target_len
    ).reshape((-1, 1))
    
    # print(recovered_audio.shape)
    # np.save("./recovered.npy", recovered_audio)
    # print(args.out_audio)
    #with open(args.out_audio, 'wb') as of:
    sf.write(args.out_audio, recovered_audio, 24000)
