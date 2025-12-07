# Echo-TTS

> Fork: Echo TTS Streaming API — adds a FastAPI server in `api_server.py` that serves `/v1/audio/speech` with streaming PCM output. It keeps upstream behavior but layers chunked text handling, configurable sampling defaults, and runtime switches via `ECHO_*` env vars.

## Running

Run:

```
docker compose up -d
```

Then review the logs, waiting for it to say that it's serving on port 8004.

Then, run this to test:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  --output out.mp3 \
  -d '{
    "input": "Hello. How are you today?",
    "model": "echo-tts",
    "voice": "Scarlett-Her",
    "stream": false,
    "seed": 0,
    "extra_body": {}
  }' && xdg-open out.mp3
```

### Open-WebUI:

Configure Open-WebUI's `Admin->Audio->Speech` section as follows:

Text-to-speech Engine: `OpenAI`
URL: `http://localhost:8004/v1`
Password/token/auth: `unused` (doesn't matter, literally unused)
TTS Voice: `Scarlett-Her`

TTS Model: `echo-tts`

Additional Parameters:
```
{
  "response_format": "mp3",
  "stream": false
}
```

### Server Environment Flags
- `ECHO_MODEL_REPO` (default `jordand/echo-tts-base`) selects the main model; `ECHO_FISH_REPO` (default `jordand/fish-s1-dac-min`) selects the decoder.
- `ECHO_DEVICE` / `ECHO_FISH_DEVICE` (default `cuda`) pick devices; set to `cpu` to avoid GPU requirements. `ECHO_MODEL_DTYPE` (default `bfloat16`) and `ECHO_FISH_DTYPE` (default `float32`) control dtypes.
- `ECHO_COMPILE` (default `0`) toggles `torch.compile` for the main model; `ECHO_COMPILE_AE` (default `1`) separately compiles the decoder; `ECHO_COMPILE_LORA_ONLY` is ignored when LoRA is unused.
- Cache/logging: `ECHO_CACHE_DIR` (default `/tmp`) and `ECHO_CACHE_VERSION` label saved compile artifacts; `ECHO_CACHE_SPEAKER_ON_GPU` (default `0`) caches speaker latents per device; `ECHO_DEBUG_LOGS` (default `0`) enables verbose timing/debug prints.
- Chunking/text defaults: `ECHO_CHUNKING` (default `1`), `ECHO_CHUNK_CHARS_PER_SECOND` (default `14`), `ECHO_CHUNK_WORDS_PER_SECOND` (default `2.7`), `ECHO_NORMALIZE_EXCLAMATION` (default `1`) normalizes `!` (single -> `.`, multiple -> `!`).
- Reference audio handling: `ECHO_MAX_SPEAKER_LATENT_LENGTH` (default `6400`), `ECHO_FOLDER_SUPPORT` (default `1` to allow folder prompts), `ECHO_WARMUP_VOICE` and `ECHO_WARMUP_TEXT` seed optional compile warmup.
- Optional dependency: ffmpeg (on PATH) is required for `response_format='mp3'`; when present, non-stream defaults to MP3, otherwise WAV.
- Performance presets (streaming only): `ECHO_PERFORMANCE_PRESET` (default `default`) sets streaming sampler defaults: `default` uses `block_sizes=[32, 128, 480]` / `num_steps=[8, 15, 20]`; `low_mid` keeps those blocks with `num_steps=[8, 10, 15]`; `low` uses `block_sizes=[32, 64, 272, 272]` and `num_steps=[8, 10, 15, 15]`. Unknown values fall back to default with a warning; non-streaming uses its own steps.
- Non-streaming steps: `ECHO_NUM_STEPS_NONSTREAM` (default `20`) controls the fixed non-stream sampler steps (recommended range 10–40); block size stays `640` by default unless overridden via request.
- Note: enabling `torch.compile` (model and/or decoder) can increase peak VRAM; disable `ECHO_COMPILE`/`ECHO_COMPILE_AE` if memory is tight.

### Performance / VRAM notes
- Quick presets (streaming): set `ECHO_PERFORMANCE_PRESET=low_mid` to reduce steps or `ECHO_PERFORMANCE_PRESET=low` to also shrink blocks; both lower compute/VRAM at some quality cost. Non-streaming always defaults to 20 steps unless you set `ECHO_NUM_STEPS_NONSTREAM` (10–40 recommended).
- Lower-end GPUs: prefer `ECHO_PERFORMANCE_PRESET=low_mid` (fewer streaming steps) or `ECHO_PERFORMANCE_PRESET=low` (smaller blocks + fewer steps) instead of manual step tweaks.
- Compile vs presets: with `ECHO_COMPILE=1` you may be able to keep the higher (default) preset while staying real-time, but it raises peak VRAM; if memory is tight, turn compile off before lowering presets.
- VRAM reduction: set `ECHO_FISH_DTYPE=bfloat16` (or `bf16`) to run the decoder in bf16 at a small quality cost.
- Disable compile to save memory: set `ECHO_COMPILE=0` (model) and `ECHO_COMPILE_AE=0` (Fish AE, which defaults to compiled) if VRAM is constrained; expect slower generations.

# Original README

A multi-speaker text-to-speech model with speaker reference conditioning. See the [blog post](https://jordandarefsky.com/blog/2025/echo/) for technical details.

**Model:** [jordand/echo-tts-base](https://huggingface.co/jordand/echo-tts-base) | **Demo:** [echo-tts-preview](https://huggingface.co/spaces/jordand/echo-tts-preview)

## Responsible Use

Don't use this model to:
- Impersonate real people without their consent
- Generate deceptive audio (e.g., fraud, misinformation, deepfakes)

You are responsible for complying with local laws regarding biometric data and voice cloning.

## Installing as a python package / executable

```
uv build
uv tool install dist/*.whl
```

Then run as `uv tool run echo-tts-api`

Requires Python 3.10+ and a CUDA-capable GPU with at least 8GB VRAM.

## Tips

### Adding voices

You can condition on up to 5 minutes of reference audio, but shorter clips (e.g., 10 seconds or shorter) work well too. 22khz mono 16-bit audio is suggested.

## License

Code in this repo is MIT‑licensed except where file headers specify otherwise (e.g., autoencoder.py is Apache‑2.0).
    
This is based on the original engine code from https://jordandarefsky.com/blog/2025/echo/.

Regardless of the model license, audio outputs are CC-BY-NC-SA-4.0 due to the dependency on the Fish Speech S1-DAC autoencoder, which is CC-BY-NC-SA-4.0.

The Echo-TTS weights are under CC-BY-NC-SA-4.0.

For included audio prompts, see `audio_prompts/LICENSE`. Scarlett*.mp3 and Jarvis.wav would fall under fair use.

## Citation

```bibtex
@misc{darefsky2025echo,
    author = {Darefsky, Jordan},
    title = {Echo-TTS},
    year = {2025},
    url = {https://jordandarefsky.com/blog/2025/echo/}
}
```
