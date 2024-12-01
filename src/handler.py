import runpod
import requests
import os
import torch
import torchaudio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperForConditionalGeneration,
    pipeline
)


def download_file(url, local_filename):
    """Helper function to download a file from a URL."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def get_speech_segments(audio_path):
    """Use Silero VAD to get speech segments from the audio."""
    # Load the VAD model
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    (get_speech_ts,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    # Read the audio
    wav, sample_rate = torchaudio.load(audio_path)
    wav = wav.squeeze()

    # Get speech timestamps
    speech_timestamps = get_speech_ts(
        wav, vad_model, sampling_rate=sample_rate)
    return speech_timestamps, sample_rate


def save_speech_segments(speech_timestamps, wav, sample_rate, output_path):
    """Save the speech segments to a new audio file."""
    import torchaudio

    speech_wav = []
    for segment in speech_timestamps:
        start = segment['start']
        end = segment['end']
        speech_wav.append(wav[int(start):int(end)])

    if speech_wav:
        speech_wav = torch.cat(speech_wav)
        torchaudio.save(output_path, speech_wav.unsqueeze(0), sample_rate)
        return True
    else:
        return False


def run_whisper_inference(audio_path, chunk_length, batch_size, language, task):
    """Run Whisper model inference on the given audio file."""
    model_id = "openai/whisper-large-v2"
    torch_dtype = torch.float16
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_cache = "/cache/huggingface/hub"
    local_files_only = True

    # Load the model, tokenizer, and feature extractor
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        cache_dir=model_cache,
        local_files_only=local_files_only,
    ).to(device)
    tokenizer = WhisperTokenizerFast.from_pretrained(
        model_id, cache_dir=model_cache, local_files_only=local_files_only
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_id, cache_dir=model_cache, local_files_only=local_files_only
    )

    # Initialize the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Adjust generate_kwargs
    generate_kwargs = {
        "task": task,
        "language": language or "he",
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "repetition_penalty": 1.2,
        "suppress_blank": True,
    }

    # Run the transcription
    outputs = pipe(
        audio_path,
        chunk_length_s=chunk_length,
        batch_size=batch_size,
        generate_kwargs=generate_kwargs,
        return_timestamps="word",
    )

    # Process outputs to get sentences with timestamps
    sentences = []
    current_sentence = ""
    current_start = None

    for word_info in outputs['chunks']:
        word = word_info['text']
        start = word_info['timestamp'][0]
        end = word_info['timestamp'][1]

        # Skip non-speech words (if any)
        if not word.strip():
            continue

        if current_start is None:
            current_start = start

        current_sentence += word + " "

        # Segment sentences based on punctuation
        if any(punct in word for punct in ['.', '!', '?']):
            sentences.append({
                'sentence': current_sentence.strip(),
                'start': current_start,
                'end': end
            })
            current_sentence = ""
            current_start = None

    # Add the last sentence if any
    if current_sentence:
        sentences.append({
            'sentence': current_sentence.strip(),
            'start': current_start,
            'end': end
        })

    return sentences


def handler(job):
    job_input = job['input']
    audio_url = job_input.get("audio")
    chunk_length = job_input.get("chunk_length", 30)
    batch_size = job_input.get("batch_size", 8)
    language = job_input.get("language", "he")
    task = job_input.get("task", "transcribe")

    if audio_url:
        # Download the audio file
        audio_file_path = download_file(audio_url, 'downloaded_audio.wav')

        # Use VAD to get speech segments
        speech_timestamps, sample_rate = get_speech_segments(audio_file_path)
        if speech_timestamps:
            # Save speech segments to a new audio file
            speech_audio_path = 'speech_audio.wav'
            wav, _ = torchaudio.load(audio_file_path)
            wav = wav.squeeze()
            save_speech_segments(speech_timestamps, wav,
                                 sample_rate, speech_audio_path)
        else:
            return "No speech detected in the audio."

        # Run Whisper model inference
        result = run_whisper_inference(
            speech_audio_path, chunk_length, batch_size, language, task)

        # Cleanup: Remove the downloaded files
        os.remove(audio_file_path)
        os.remove(speech_audio_path)

        return result
    else:
        return "No audio URL provided."


runpod.serverless.start({"handler": handler})
