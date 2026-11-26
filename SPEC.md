# SubBuild Studio — Technical Specification (`SPEC.md`)

> Architecture, technology stack and design decisions of SubBuild Studio as it is built today.
> For setup, usage, every setting and the full endpoint list see [`README.md`](README.md); this document
> explains **how the system is put together and why**. It was rewritten from the original build
> specification (phases 1–4) to match the current code.

## 1. Overview

**SubBuild Studio** is a local application that turns a video file into finished subtitles:

1. **Transcribes** the audio on the user's own machine (faster-whisper, word-level timestamps, GPU or CPU).
2. **Refines** Whisper's output into broadcast-style subtitles (sentence re-segmentation, line limits, reading speed).
3. **Translates** via **Google Gemini, OpenAI, Anthropic Claude, or Ollama (Local / Cloud)** (context-aware batches, glossary, first-class Hebrew/RTL support) — optional.
4. **Re-times a subtitle file the user already has** against the video's speech and on-screen text — an alternative to 1–3.
5. Offers an **editing studio** (synced player, inline edit, add/move/re-time lines, quality checks) and **SRT/VTT export**.

It runs from source in two modes: as a **desktop window** (`run.bat` / `./run.sh --window`) and as a **local web app** (`run.bat --browser` / `./run.sh`),
with a built-in Git & Release auto-updater (`run.bat` / `./run.sh` pre-launch check + in-app Sidebar/Settings UI).
**Recommended usage is the desktop window** (rather than a regular browser) so that dragged-and-dropped or browsed files resolve directly to their local disk path without uploading or copying to temp. Nothing leaves the machine except the text
and frames explicitly sent to your configured cloud AI provider (or stays 100% offline with local Ollama); the video itself is never uploaded anywhere.

### Core engineering rules

1. **English-only code, comments, docstrings, logs and commit messages.** (The UI itself is bilingual: English and Hebrew.)
2. **Pipeline + provider architecture.** Work is a sequence of stages; transcription and multi-provider translation sit behind abstract
   interfaces (`BaseTranscriber`, `BaseTranslator`, `providers/llm/client.py`) so engines can be swapped or faked in tests.
3. **Never block the event loop.** `ffmpeg`, Whisper and synchronous provider SDK calls run in worker threads (`asyncio.to_thread`), while HTTP AI providers (`OpenAI`, `Anthropic`, `Ollama`) use async `httpx` clients; progress reaches the browser over a WebSocket.
4. **Zero-copy local files.** Videos are 1–20 GB: the unified `MediaDropzone` lets the user drag & drop a video, browse via the native file dialog, or paste a path on disk. In the desktop window, dragged and browsed files resolve directly to their local path (nothing is copied, and *Save next to video* remains available); only when running inside an external browser (where browser sandboxing hides disk paths) does drag-and-drop fall back to uploading a copy into `storage/uploads` (owned by the app and deleted with its project).
5. **Typed multi-provider configuration & key vault (`google-genai`, `httpx`).** Env defaults come from `backend/.env` via `pydantic-settings`;
   the Settings screen manages multiple named keys per provider (`storage/provider_keys.json`), discovers live available models, and overrides active per-step choices in `storage/settings.json` (and can persist `GEMINI_API_KEY` to `.env` via `PUT /api/settings/api-key`). Per-job Gemini session overrides typed in Ingest stay in memory for that run only.
6. **Safe by default for a local server.** The API can read any path on the PC, so it binds to `127.0.0.1`, requires a
   per-launch session token (`app/core/auth.py`: HttpOnly cookie or `X-SubBuild-Token` header), uses an explicit
   CORS allow-list, a trusted-host check (against DNS rebinding), a cross-site request guard (against CSRF) and
   anti-framing headers on every response (`app/core/security_headers.py`, against clickjacking).
7. **Nothing is lost.** Results are saved as soon as they exist (after refinement and after every translated batch); failures
   are flagged on the line, never silently dropped; jobs interrupted by a restart are recovered.

---

## 2. Technology stack

| Layer | Technology | Role |
|---|---|---|
| Backend framework | **Python 3.11+ (3.12 in `run.bat`), FastAPI, Uvicorn** | REST API, WebSocket, static hosting of the built frontend |
| Persistence | **SQLModel / SQLAlchemy on SQLite** (`storage/subbuild.db`) | Jobs and subtitle segments; additive column migration at startup |
| Configuration | **pydantic-settings** (`.env`) + `storage/settings.json` + `storage/provider_keys.json` | First-run defaults, multi-provider key vault, per-step Settings |
| Speech-to-text & Audio Conditioning | **faster-whisper** (CTranslate2) + **Silero VAD**, optional **WhisperX** (`wav2vec2` phoneme alignment), **Smart Audio Map** (`numpy` FFT scanner) + optional **Demucs / audio-separator** | Local transcription, word timestamps, selective window conditioning (whisper gain, localized Demucs vocal isolation, comms bandpass), speech maps, language detection |
| Speaker Diarization & Pitch | **WeSpeaker CAM++ ONNX** (`onnxruntime`) + **F0 pitch estimator** (`voice_gender.py`) | Zero-token local speaker clustering, long-film (`>= 45m`) oversized cluster re-split, speaker gender from raw audio pitch |
| GPU runtime | **NVIDIA cuBLAS / cuDNN wheels** (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) | CUDA FP16 without a system CUDA install; CPU INT8 fallback |
| Media | **ffmpeg / ffprobe** (async subprocesses) | Probing, audio extraction and filtering, frame sampling (GPU-accelerated), subtitle-track extraction |
| Cloud & Local AI | **Google Gemini** (`google-genai`), **OpenAI**, **Anthropic Claude**, **Ollama** (`httpx`) | Glossary pass, translation, on-screen text (vision), sync line matching, speaker gender detection |
| Local AI Optimization | **Ollama** (local LLM), **Pillow** (edge & dHash) | Local gender pre-tagging, frame deduplication & blank-frame pruning in Cost-Saver Mode |
| Frontend | **React 19, TypeScript 6, Vite 8, Tailwind CSS 4, Zustand 5, lucide-react** | Single-page studio; typed REST client; WebSocket listener inside the job store |
| Lint / build | **oxlint**, `tsc -b` + `vite build` | Zero-error lint, strict TypeScript build, 85 Vitest tests across 16 spec files |
| Desktop shell | **pywebview** (Edge WebView2), **pystray + Pillow** (tray icon) | Own app window and tray menu from source (`packaging/launcher.py`) |
| Auto-Updater | **Git & GitHub Releases** (`VERSION`, `backend/app/core/updater.py`) | Semver release tag (`vX.Y.Z`) update checks on launch (`run.bat`/`run.sh`) and in-app UI |
| Tests | **pytest, pytest-asyncio, httpx, Vitest, Java 21 + Selenium 4 + RestAssured** | 820 Pytest + 85 Vitest + 53 Java E2E/API tests |

---

## 3. System architecture

```text
 ┌──────────────────────────── Browser tab  or  pywebview window (WebView2) ───────────────────────────┐
 │  React SPA:  Ingest ─ Live monitor ─ Studio (player + table) ─ Settings ─ Help & Q&A                 │
 │  Zustand stores (jobs, settings, ui)   typed REST client (services/api.ts)   WebSocket listener       │
 └───────────────▲──────────────────────────────────────┬───────────────────────────────────────────────┘
                 │ REST /api/*   (Range streaming for video)      │ WS /ws/jobs/{id}: progress, segments
 ┌───────────────┴──────────────────────────────────────▼───────────────────────────────────────────────┐
 │ FastAPI app  (CORS allow-list, TrustedHost, routers, serves frontend/dist)                           │
 │   routes: system · media · jobs · export · settings · glossary · storage                              │
 │                                    │                                                                  │
 │                       PipelineOrchestrator  (one asyncio task per job)                                │
 │        GPU semaphore · cancel flag · save-as-you-go · usage/cost tracking · per-job log               │
 │                                    │                                                                  │
 │   Stages:  Probe → Audio → [SmartAudio/Vocals] → Transcribe → [Diarize] → [On-screen] → Refine → [Translate] (full)  │
 │            Probe → Audio → Transcribe → [On-screen] → Sync                                                   (sync)  │
 │            Translate only                                                                               (translate)  │
 │                                    │                                                                  │
 │   Providers:  BaseTranscriber ── FasterWhisperTranscriber · WhisperXProvider                          │
 │               BaseTranslator  ── LLMTranslator (Gemini Flex · OpenAI · Anthropic · Ollama) · Noop     │
 └────────┬──────────────────────────┬────────────────────────────┬───────────────────────────┬─────────┘
          │ SQLite (SQLModel)        │ ffmpeg / ffprobe           │ faster-whisper (GPU/CPU)  │ Multi-Provider AI / Ollama
          ▼                          ▼                            ▼                           ▼
   storage/subbuild.db        audio, frames, tracks         models in the HF cache        text + frames only
```

### 3.1 Backend layout (`backend/app/`)

| Package | Contents |
|---|---|
| `main.py` | App factory: lifespan (DB init, crash recovery, housekeeping), middleware (TrustedHost, CORS, `CrossSiteRequestGuard`), routers, WebSocket endpoint, static hosting of `SUBBUILD_FRONTEND_DIST` / `frontend/dist` |
| `api/` | One router per area (see §6): `routes_system`, `routes_media`, `routes_jobs`, `routes_export`, `routes_settings`, `routes_glossary`, `routes_storage` |
| `core/` | `config` (settings), `request_guard` (CSRF / cross-site WebSocket guard), `database` (engine + additive migration), `hardware` (GPU/VRAM detection, CUDA DLL paths), `model_catalog` (Gemini model metadata & historical pricing rules), `provider_catalog` (multi-provider catalog, strict capability/key filtering, episode cost estimates, GPU/VRAM-aware best-value auto-recommendation), `provider_keys` (encrypted/masked key vault in `storage/provider_keys.json` + `.env` compatibility), `model_discovery` (live Gemini/OpenAI/Anthropic/Ollama `/api/show` probing & snapshot deduplication), `user_settings`, `glossary_profiles`, `ws_manager`, `job_logging`, `housekeeping`, `storage_cleanup`, `ollama_setup` (Ollama lifecycle/install), `compat` (PyAV/faster-whisper shim) |
| `models/` | `db_models` (tables, `JobStatus`, `SegmentType`), `schemas` (Pydantic request/response models) |
| `pipeline/` | `orchestrator`, `job_report` (per-job log: settings at start, result and usage at the end), `context` (state passed between stages), `stages/*` (thin stage classes incl. `vocal_isolation_stage` and `diarize_stage`), `diarization` (WeSpeaker CAM++ ONNX clustering, long-film `>= 45m` cluster re-split & `assess_diarization` quality guard), `speaker_match` (conversational & AI speaker-to-character matching), `voice_gender` (F0 pitch gender estimation), `addressee` (confidence-scored listener gender/plurality resolver with scene continuity), `waveform` (50 peaks/s amplitude extraction), `segmentation` (Whisper words → subtitle blocks, used by Refine), `timeline_merge` (COMMS tagging, ON_SCREEN and FOREIGN merging, used by Refine), `embedded_tracks` (pick/extract forced and dialogue subtitle tracks), `frame_sampling` (frames, thumbnails, frame selection, subtitle crops), `onscreen_vision` (vision prompts + multi-provider Vision AI calls), `onscreen_merge` (ON_SCREEN segment helpers, detection merging and filtering), `missing_speech` (Whisper supplement for embedded/synced subtitles), `onscreen_prefilter` (edge & dHash frame filter with steady-state retention), `credits_detect` (smart end-credits detection & mid/post-credits scene preservation), `gender_pretag` (local Ollama pre-tagger), `glossary_pruner` (batch glossary pruner), and the algorithms below |
| `services/` | `smart_audio_map` (vectorized acoustic timeline scanner & selective window conditioner with 80 ms raised-cosine crossfades), `audio_isolation` (Demucs / `audio-separator` vocal extraction, OOM fallback, deterministic `storage/vocals/` caching) |
| `providers/` | `transcription/` (`BaseTranscriber`, `faster_whisper`, `whisperx_provider`), `translation/` (`BaseTranslator`, Gemini, Noop), `llm/client.py` (unified LLM/Vision client for Google Gemini, OpenAI, Anthropic, and Ollama), `gemini_calls` (Flex tier wrapper with 2-min timeout, temporary cooldown & single-attempt re-probe) |
| `utils/` | SRT/VTT formatter, RTL fixer, speaker turn splitter & SDH tag cleaner (`speaker_split`), subtitle file parser, language helpers, text-break rules |

### 3.2 Pipeline stages (`pipeline/stages/`)

| Stage (`JobStatus`) | What it does |
|---|---|
| **Probe** (`PROBING`) | `ffprobe` the file: duration, chapters (`-show_chapters` for end-credits / post-credits markers), audio streams (the user picks one), embedded subtitle tracks (dialogue vs. forced/signs) |
| **Audio** (`EXTRACTING_AUDIO`) | If an embedded text dialogue track is selected (`use_embedded_subtitles`) and `supplement_with_whisper` is off, extracts that subtitle stream directly via FFmpeg and skips audio extraction unless needed for foreign speech or diarization. Otherwise extracts mono 16 kHz WAV (downmixing surround with centre-channel dialogue boost; explicit *Comms boost* `on` adds a full-file speech-band chain: band-pass, FFT denoise, compressor, `dynaudnorm`) |
| **Audio Conditioning / Vocal Isolation** (`VOCAL_ISOLATION`, optional) | Preserves the clean extracted WAV in `ctx.raw_audio_wav_path` for diarization and F0 pitch. When `comms_boost_mode="auto"` (Smart Audio) or `vocal_isolation_mode="auto"` is active, `smart_audio_map.py` scans the 16 kHz timeline in ~1–2s to locate `whisper` and `music_masked` windows, applying localized soft gain or selective Demucs separation only on those windows while leaving the rest of the soundtrack 100% untouched (`OFF`) and blending edges with 80 ms raised-cosine crossfades. When `vocal_isolation_mode="on"`, runs full-file Demucs / `audio-separator` vocal isolation with source caching in `storage/vocals/` |
| **Transcribe** (`TRANSCRIBING`) | Uses extracted embedded subtitles when available; if `supplement_with_whisper` is enabled, also runs Whisper and `missing_speech.py` to fill uncovered spoken gaps. Full transcription jobs run with either **`faster_whisper`** (Silero VAD, word timestamps, optional `high_speech_sensitivity` thresholds, batched GPU inference with sequential fallback on OOM, *start alignment* `word_snap`, *missed-speech recovery* `gap_recovery`) or **`whisperx`** (batched VAD ASR + `wav2vec2` phoneme forced alignment + phantom interjection filtering `filter_phantom_interjections`), plus *foreign-speech detection* (`foreign_speech`). User `initial_prompt` / context glossary terms seed ASR recognition. In Cost-Saver Mode (or after WhisperX steps), models are unallocated from GPU memory immediately |
| **Diarize** (`DIARIZING`, optional) | Runs local zero-token WeSpeaker CAM++ ONNX voiceprint clustering (`diarization.py`) over `ctx.raw_audio_wav_path`, re-splits oversized merged clusters on long films (`>= 45 min`), estimates each speaker's gender from median F0 voice pitch (`voice_gender.py`), and links voice clusters to character names (`speaker_match.py` with a 16k output token budget) |
| **On-screen text** (`SCANNING_ONSCREEN`, optional) | Uses an embedded forced/"signs" subtitle track if present; otherwise samples frames in the background while Whisper runs, scaled by `onscreen_sensitivity` (1=Strict ~3s/360 frames, 2=Balanced ~2s/660 frames, 3=Broad ~1s/1,200 frames + scene cuts). Frame extraction is GPU-accelerated (`-hwaccel auto` and `-discard nokey`). When `skip_end_credits` is enabled, `credits_detect.py` detects dark scrolling credit-roll spans from `thumbs.raw` and chapters, stores `ctx.visual_credit_spans`, skips scrolling credit frames (while keeping static stinger cards and mid/post-credits scenes), and prevents `foreign_spans` from scanning scrolling credits. In Cost-Saver Mode, frames are pre-filtered locally (Pillow edge detection + 64-bit dHash, keeping steady-state frames so sign disappearance isn't truncated). Also reads burned-in subtitles for foreign passages |
| **Refine** (`REFINING`) | Flatten words → remove repetition loops → re-group by sentence → multi-speaker turn splitting & optional SDH speaker/sound-tag cleanup (`speaker_split.py`, `remove_sdh_speakers`) → split at clause boundaries (42 chars/line, 2 lines) → timing pass (no overlaps, ≥ 2-frame gaps, target reading speed) → merge `[ON_SCREEN]` cues into the timeline (`allow_onscreen_overlap`, ≥ 3.0s minimum reading duration) → filter end-credits cast/crew cards and closing song lyrics (`filter_credits_segments` when `skip_end_credits=True`) while preserving mid/post-credits scenes and narrative stinger cards |
| **Translate** (`TRANSLATING`, optional) | Glossary pass over the whole transcript (merged with user `initial_prompt` / context glossary pairs), diarization quality guard (`assess_diarization`), contextual speaker gender refinement, confidence-scored listener resolution (`addressee.py`), and batched translation (model-aware batch size: 35 lines for capable cloud models, tuned smaller batches for local/compact models) with static system/glossary prefix caching, per-stage thinking budgets, sliding context, and strict 1-to-1 index mapping. Spoken lines (`DIALOGUE`/`COMMS`/`FOREIGN`) and `[ON_SCREEN]` signs are translated in separate passes with dedicated prompts. Lines contradicting a verified listener are auto-fixed once (`ADDRESSEE_AUTOFIX`); per-line retry; failures flagged; dynamic glossary pruning per batch; local Ollama gender pre-tagging in Cost-Saver Mode |
| **Sync** (`SYNCING`, sync jobs) | Re-times the user's subtitle file (and optionally supplements missing spoken lines with Whisper when `supplement_with_whisper` is enabled), see §4.4 |

A job moves through `IDLE/QUEUED → … → COMPLETED | FAILED | CANCELLED`. `QUEUED` is the wait for the GPU slot
(`MAX_CONCURRENT_TRANSCRIPTIONS`, default 1). On startup `recover_interrupted_jobs()` repairs jobs left mid-run by a crash:
jobs that already have subtitles become usable (`COMPLETED` with a warning, untranslated lines flagged), others `FAILED`.

### 3.3 Real-time protocol (`WS /ws/jobs/{job_id}`)

One JSON envelope: `{ event, job_id, stage?, progress_percent, speed_factor?, eta_seconds?, segment?, segments_batch?, message? }`.

| `event` | Meaning |
|---|---|
| `STAGE_PROGRESS` | Stage, percent, speed factor, ETA, optional status message |
| `SEGMENT_CREATED` | A transcribed segment, streamed while Whisper runs |
| `SEGMENTS_REFINED` | The final, editable subtitles (with database ids) replace the raw Whisper segments |
| `BATCH_TRANSLATED` | A translated batch; each batch is also saved to the database |
| `JOB_COMPLETED` / `JOB_FAILED` / `JOB_CANCELLED` | Terminal events |

The frontend store reconnects automatically and re-fetches the job on reconnect, so the database — not the socket — is the source of truth.

---

## 4. Key algorithms and design decisions

### 4.1 Transcription, Smart Audio Map and timing quality
- **Hardware:** CUDA FP16 when an NVIDIA GPU is found, CPU INT8 otherwise (`WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`). In the frozen build the CUDA wheels' `bin` folders are added to the DLL search path (`core/hardware.py`).
- **Dual ASR Providers (`faster_whisper` & `whisperx`):** `FasterWhisperTranscriber` is the default lightweight CTranslate2 engine (including `ivrit-ai/whisper-large-v3-turbo-ct2` for Hebrew). `WhisperXProvider` (`providers/transcription/whisperx_provider.py`, lazily loaded when installed) runs batched VAD Whisper ASR followed by `wav2vec2` phoneme forced alignment, interpolates unaligned digit/symbol timestamps, drops isolated low-alignment breath hallucinations (`filter_phantom_interjections`), and releases PyTorch CUDA VRAM after every step.
- **Smart Audio Map & Selective Window Conditioning (`services/smart_audio_map.py`, `services/audio_isolation.py`, `stages/vocal_isolation_stage.py`):**
  - Instead of running heavy DSP or Demucs across a full 2-hour movie, `analyze_audio_timeline()` scans the 16 kHz WAV in 10-minute chunks (~1–2s on CPU, `<50 MB` RAM) using STFT band power shares, spectral flatness, and sub-frame envelope variance (`sub_rms_cv`) calibrated to the soundtrack's 88th-percentile dialogue loudness.
  - Identifies localized windows by kind (`comms` > `music_masked` > `whisper`): applies soft linear speech gain (up to `+9 dB`) on `whisper` windows, runs **Demucs (`htdemucs`)** vocal separation *only* on `music_masked` windows (capped at 24 windows / 600s), and applies radio bandpass + spectral hiss reduction on explicit `comms` windows.
  - All audio outside detected windows remains **100% untouched (`OFF`)**, window boundaries are joined with an 80 ms raised-cosine crossfade, and `ctx.raw_audio_wav_path` is preserved untouched so WeSpeaker diarization and F0 voice-gender estimation never see separation or compression artifacts.
- **Embedded subtitle extraction & Whisper supplement (`use_embedded_subtitles`, `supplement_with_whisper`, `missing_speech.py`):** When an embedded text dialogue track is selected, FFmpeg extracts it directly. If `supplement_with_whisper` is also enabled, Whisper transcribes the audio and `missing_speech.py` identifies uncovered spoken gaps (filtering out hallucinations and overlapping cues) and weaves them into the subtitle timeline.
- **High speech sensitivity (`high_speech_sensitivity`):** Relaxes VAD and decoding thresholds (`no_speech_threshold`, `log_prob_threshold`, `speech_pad_ms`) to capture quiet whispers and speech partially masked by music or effects.
- **Hallucination control:** Silero VAD, `condition_on_previous_text=False`, repetition-loop removal, strict checks on recovered text.
- **Word start alignment (`word_snap`):** Whisper's timestamps ignore silence, so a word after a pause can be stamped early. Silero VAD finds the real speech onset; starts are only ever moved *later*, and only across silent gaps ≥ 0.4 s.
- **Missed-speech recovery (`gap_recovery`):** VAD misses shouted speech under gunfire/music; Faster-Whisper re-listens *without* VAD to uncovered stretches and keeps only lines that pass hallucination checks.
- **Foreign passages (`foreign_speech`):** per-chunk language detection (encoder only) finds speech in another language than the rest; the text comes from burned-in subtitles (frame reading) or a Whisper re-transcription in that language.
- **Re-check timing (`retime`, `speech_regions`):** timing-only correction of finished subtitles against a fresh speech map; text and on-screen cues are never touched; cached under `storage/regions`; the studio offers *Undo*.

### 4.2 Subtitle refinement and export
- Linguistic break rules live in `utils/text_rules.py`; multi-speaker turn splitting and SDH speaker/sound-tag cleanup (`remove_sdh_speakers`) in `utils/speaker_split.py`; formatting (balanced line splitting, SRT/VTT) in `utils/srt_formatter.py`.
- Limits are configurable: `MAX_CHARS_PER_LINE` 42, `TARGET_CPS` 17, `MAX_CPS_WARNING` 20, duration 1–7 s, minimum gap ≈ 2 frames.
- **RTL (Hebrew/Arabic):** text is stored clean; directional marks (`U+200F`) are added only at export so `? ! .` and dialogue dashes land on the right side in VLC/Plex/HTML5 players; SRT is written with a UTF-8 BOM.
- Segment types: `DIALOGUE`, `COMMS` (radio traffic), `ON_SCREEN` (titles/signs, bracketed), `FOREIGN` (other-language speech).
- **On-Screen / Dialogue Overlap & Adequate Reading Time (`allow_onscreen_overlap`):** On-screen chyrons/titles can display concurrently with bottom dialogue subtitles without clipping subsequent speech lines (`allow_onscreen_overlap=True` by default). On-screen cues enforce a minimum 3.0-second reading duration (or `chars / 15 cps` for longer titles). Manual segment insertion in the studio distinguishes dialogue from on-screen lines to prevent collision false-positives.
- **On-Screen Subtitle Positioning (`onscreen_position`):** Configurable per-job in post-video Ingest (`PipelineConfig`, alongside `convert_to_metric`, `remove_sdh_speakers`, and `skip_end_credits`) and live inside the Studio toolbar (`PATCH /api/jobs/{id}/settings`: `top_center`, `middle_center`, `middle_right`, `middle_left`, `top_right`, `top_left`). Serialized using standard SubStation Alpha tags `{\anX}` in SRT (supported by VLC, MPC-HC, MPV, PotPlayer) and positioning cue settings (`line:8%`, `line:50% position:90%`, etc.) in WebVTT, as well as live HTML5 video player dual-layer overlay coordinates.
- **Smart End-Credits Filtering & Post-Credits Scene Preservation (`skip_end_credits`, `pipeline/credits_detect.py`):** Combines container chapter markers (`ffprobe -show_chapters`), 64×36 grayscale thumbnail luma/contrast analysis (`detect_visual_credit_spans` on `thumbs.raw` with pure-black card fade bridging), and dialogue-vs-lyric cluster classification (`filter_credits_segments`) to strip cast/crew cards, vendor tags, and closing songs (including punctuated rap/hip-hop verses leading into choruses) in the tail of the video while preserving mid-credits and post-credits dialogue scenes and narrative stinger cards (`[THOR WILL RETURN]`, `[MEANWHILE...]`).

### 4.3 Translation & Multi-Provider AI
- **Unified multi-provider routing (`providers/llm/client.py`, `core/provider_catalog.py`):** Stages route cleanly across **Google Gemini**, **OpenAI GPT**, **Anthropic Claude**, and **Ollama (Local / Cloud)**. Users can mix and match providers across *Translation*, *Glossary pass*, and *On-screen text (vision)* (for example, local Ollama for dialogue + Google Gemini Flex or GPT-5 nano for frame OCR).
- **Multi-Provider Key Vault & Live Discovery (`core/provider_keys.py`, `core/model_discovery.py`):** Supports multiple named keys per provider stored in `storage/provider_keys.json` alongside `.env` keys. Keys trigger live discovery (`POST /api/settings/providers/refresh`), inspecting Ollama `/api/show` for native `vision` capability and parameter sizes (`param_billions`), deduplicating redundant dated OpenAI snapshots, and strictly filtering dropdowns so only capable, runnable models appear for each pipeline step.
- **Glossary pass → batches:** characters (gender, name spelling), terms and tone feed every batch; **series glossary profiles** (`storage/glossaries/`) carry them across episodes.
- **Dedicated On-Screen Translation Pass:** Spoken dialogue (`DIALOGUE`, `COMMS`, `FOREIGN`) and `[ON_SCREEN]` signs are translated in separate passes with dedicated prompts (`build_onscreen_system_prompt`), preventing sign text and overlapping dialogue from contaminating or overwriting each other.
- **Structured output** with strict index mapping; sliding `previous_context` / `lookahead_context`; missing lines retried individually.
- **Metric System Conversion (`convert_to_metric`):** When enabled, automatically converts non-metric imperial units (miles, feet, inches, yards, pounds/lbs, ounces, Fahrenheit, gallons, acres) into idiomatic metric equivalents (km, km/h, meters, cm, kg, Celsius, liters, dunams) with natural rounding for subtitle reading speed, while preserving figurative/idiomatic expressions ("give an inch", "walk a mile").
- **Per-step models, Thinking Budgets & Best-Value Auto-Selector:** *Translation*, *Glossary pass* and *On-screen text (vision)* each have their own provider, key selection, model, and (for Gemini) per-stage thinking budget, paired with model-aware batch sizing (`35` lines for capable cloud models) and static system/glossary prefix caching. The **Auto-select best value (Cost / Quality)** optimizer compares all configured keys across providers (including Google **Flex** `-50%`) to assign the cheapest strong model per stage, with an optional GPU/VRAM-aware **"Include free Local AI"** toggle that only picks verified local models (`>= 6.5B`) fitting the detected CUDA VRAM.
- **Flex tier with temporary cooldown & single-attempt re-probe (`providers/gemini_calls.py`):** Google Gemini requests at half price with a 2-minute timeout (`FLEX_TIMEOUT_MS=120000`) and asyncio safety guard (`FLEX_CLIENT_GUARD_SECONDS`); immediate fallback to Standard on `504 DEADLINE_EXCEEDED` or capacity errors (429/503) after exponential backoff; after two consecutive slow (`>60s`) or failed Flex calls, `FlexState` enters a temporary cooldown (`12` requests or `90s`) and then performs a fast single-attempt re-probe on Flex so discounted pricing resumes automatically when Google capacity recovers.
- **Cost:** every call records exact input/output/thinking tokens and dollars per step (`Job.usage_json`); `POST /api/settings/estimate` predicts cost across all providers before running.
- **Row-level tools:** re-translate selected rows (`RowTranslateDialog`) using any active translation-capable provider/model with speaker/addressee gender and a free-text note; *Detect from video* asks the Vision AI provider to infer gender from frames; a deterministic **quality check** (`pipeline/qa.py`, no model call) flags untranslated Latin words, glossary mismatches, pronoun/verb gender errors and 2nd-person forms that contradict a line's confidently resolved listener (`pipeline/addressee.py`; each row stores its listener in `addressee_json` and shows it as an editable *Speaking to* tag).

### 4.4 Syncing an existing subtitle file (`sync_*`)
Input: video + the user's own SRT / WebVTT / ASS (`utils/subtitle_parser.py` reads UTF-8/16, Windows-1255 and -1252; formatting is stripped, top-positioned cues marked as probable signs). **Text is never changed — only timing.**

1. **Speech map:** Silero VAD over the audio plus Whisper word times.
2. **Reference timing from the video itself** when available: an embedded text subtitle track, else subtitles burned into the picture (`sync_burned.py` — local frame analysis of the bottom band with ffmpeg, no OCR; used only if it fits the speech).
3. **Global mapping** `t → scale·t + offset` by cross-correlating "a subtitle is showing" with "someone is speaking", trying the usual frame-rate ratios (25 ↔ 23.976/24 …) (`sync_align.py`).
4. **Per-cue offsets by dynamic programming (Viterbi)** with split penalties, so cuts (recaps, removed scenes) are found; an onset bonus pulls cues to speech starts.
5. **Text anchors** (`sync_text.py`): same-language files are matched to Whisper words locally (difflib); other-language files are matched by AI line-to-phrase batches (the *Translation* model, optional). Anchors are evidence, not orders — neighbours outvote a wrong one.
6. **Second alignment pass** with the anchors; low-confidence lines surface under *Only problems*; the job stores a report (`sync_report_json`: offset, frame rate, cuts, method per line).
7. The result is an ordinary project (edit, re-check timing, export). A file in another language becomes the translation, with the audio's text as the original.

### 4.5 Editing operations & Virtualized Studio Table
All edits go through the REST API and renumber lines transactionally: single/bulk patch, **add a line** (`POST /segments`, overlap → `409`, timecode `HH:MM:SS,mmm` shared with the UI), **delete a line** (`DELETE /segments/{segment_id}` with renumbering), **bulk delete lines** (`DELETE /segments` with select-all and 1-click range expansion "Select from `#X` to end" / "Select from start to `#X`"), **move** (`/segments/move`, text moves and time slots stay, only the affected span is re-checked), **re-time**, **live job settings update** (`PATCH /jobs/{id}/settings` for `onscreen_position`), single/bulk project deletion (`POST /jobs/bulk-delete`, `DELETE /jobs`). In the studio, lines can be filtered by tag (`ON_SCREEN`, `COMMS`, `FOREIGN`, or any tagged cue) as well as by categorized problem filters (`ALL_PROBLEMS`, `UNTRANSLATED`, `LATIN_LEFTOVER`, `HIGH_CPS`). For 60fps responsiveness on 2,500+ line feature films, `SubtitleTable.tsx` uses windowed row virtualization (mounting only visible viewport rows + overscan buffer with top/bottom spacer rows), primitive `activeIndex` Zustand selector subscriptions, `useDeferredValue` search filtering, and quick jump to `#line` or `HH:MM:SS` timestamp.

### 4.5.1 Edit history (undo / redo / History panel)
`core/edit_history.py` + table `job_history`. Every editing route calls `history.begin()` before changing the subtitles (stores the
state before the first edit as `original`, and drops steps ahead of the cursor, so an edit after an undo starts a new branch, as in
Photoshop) and `history.record()` after committing (a zlib-compressed JSON snapshot of all the job's lines + its sha1). `jobs.history_cursor`
points at the step that matches the subtitles now; undo/redo/restore replace the job's rows with a snapshot (keeping segment ids, or giving
a new id when another job took one) and move the cursor — later steps stay until the next edit. Autosaves of the same line and field
within 90 s update one step (`coalesce_key`); a no-op edit adds nothing; a change made outside the history (an interrupted run) is
noticed by its digest and kept as an `external` step. A re-translation is one `translate` step (begun by `POST …/translate`, recorded by
the orchestrator when the run ends). Nothing is recorded while a run is changing the subtitles, and undo is refused then (`409`). Edits of
one job are serialised with a per-job lock; the newest 100 steps are kept. The studio shows Undo/Redo and the History panel in the table
toolbar (`studio/table/HistoryControls`), with `Ctrl+Z` / `Ctrl+Y` / `Ctrl+Shift+Z` by physical key (works with a Hebrew layout); undo
first waits for autosaves still in flight.

### 4.6 Cost-Saver Mode & Local AI Offloading (`cost_saver_mode`)
- **Local frame pre-filtering (`onscreen_prefilter.py`):** Pillow-based edge detection and 64-bit difference hashing (`dHash`) filter out blank/textless frames and deduplicate consecutive identical title cards (while keeping steady-state boundary frames so sign disappearance times aren't truncated) before calling Vision AI, pruning ~75–80% of Vision image tokens.
- **Whisper GPU VRAM unloading:** Faster-Whisper model is cleanly unallocated from GPU memory immediately after transcription completes, freeing VRAM for local LLMs or other GPU tasks on 8 GB cards.
- **Local gender pre-tagging (`gender_pretag.py`):** Auto-detects local Ollama (e.g. `qwen2.5:7b`) and pre-tags ambiguous pronoun lines (`you`, `I am`) with `[Speaker: M/F, To: M/F]` hints locally for Hebrew/Arabic translation, cutting cloud token overhead. Skips with zero delay if Ollama is absent.
- **Ollama setup & lifecycle management (`core/ollama_setup.py`):** Live status polling (`GET /api/settings/ollama/status`) and background one-click automated setup (`POST /api/settings/ollama/setup`) which downloads the installer on Windows/Linux/macOS, launches the daemon, and pulls `qwen2.5:7b`.
- **Dynamic glossary pruning (`glossary_pruner.py`):** Prunes full series or project glossaries down to only the characters and terms appearing in each translation batch, reducing input token overhead.
- **Live & post-run token breakdown:** Tracks input, output, thinking tokens per AI model (provider and model), local Ollama queries handled, and frames/tokens saved locally.
- **Hardware-accelerated frame sampling:** FFmpeg sampling uses `-hwaccel auto` (NVDEC / D3D11VA) and `-discard nokey` on full video files in both OCR scanning and burned-in subtitle detection (`sync_burned.py`), reducing 36-min video sampling from 20+ min to ~29 seconds.

---

## 5. Data model

**`jobs`** — one row per project: `id`, `video_path`, `file_name`, `is_upload`, `status`, `current_stage`, `progress_percent`,
`speed_factor`, `eta_seconds`, `error_message`, `warning_message`, `video_duration`, `audio_streams_json`, `selected_audio_stream`,
`transcription_provider`, `whisper_model`, `source_language`, `detected_language`, `vad_filter`, `vocal_isolation`, `vocal_isolation_mode`,
`comms_boost`, `comms_boost_mode`, `high_speech_sensitivity`, `initial_prompt`, `use_embedded_subtitles`,
`selected_subtitle_stream`, `supplement_with_whisper`, `enable_diarization`, `detect_onscreen_text`, `onscreen_sensitivity`, `allow_onscreen_overlap`,
`onscreen_position`, `enable_translation`, `convert_to_metric`, `remove_sdh_speakers`, `skip_end_credits`, `target_language`, `gemini_model`,
`context_notes`, `glossary_profile`, `glossary_json`, `speakers_json`, `diarization_json`, `usage_json`, and for sync jobs
`job_kind` (`transcribe` | `sync`), `sync_file_name`, `sync_source_json`, `sync_report_json`, plus timestamps.

**`job_history`** — `id`, `job_id` (FK, cascade delete), `action`, `details_json`, `coalesce_key`, `segment_count`, `digest`, `snapshot` (zlib JSON of all lines), `created_at`; `jobs.history_cursor` is the current step.

**`subtitle_segments`** — `id`, `job_id` (FK, cascade delete), `index`, `start_time`, `end_time`, `original_text`, `translated_text`,
`translation_failed`, `segment_type`, `confidence`, `cps`, `speaker`, `addressee_json`, `words_json`.

**Migrations:** `core/database.py` keeps an additive `_ADDED_COLUMNS` map and `ALTER TABLE … ADD COLUMN`s missing columns at startup, so
existing databases keep working across versions (no migration tool, no destructive changes).

**On disk (`storage/`, git-ignored):** `subbuild.db`, `settings.json`, `uploads/`, `audio/`, `vocals/`, `frames/`, `exports/`, `regions/`, `glossaries/`, `logs/` (`server.log` rotating 3 × 5 MB, `jobs/<job>.log` per project).
Housekeeping at startup removes stale temporary audio and abandoned uploads (`UPLOAD_RETENTION_HOURS`); the Settings screen can free space by category and never touches videos the user selected by path.

---

## 6. API surface

Interactive documentation: `http://localhost:8000/docs`. Summary:

| Area | Endpoints |
|---|---|
| System | `GET /api/system/info`, `GET /api/system/update/status`, `POST /api/system/update/check`, `POST /api/system/update/apply`, `GET /api/system/log` |
| Media | `POST /api/media/probe-path`, `POST /api/media/upload`, `GET /api/media/upload/existing`, `POST /api/media/parse-subtitles`, `GET /api/media/stream/{job_id}` (HTTP Range) |
| Jobs | `POST/GET /api/jobs` (`sync_cues` makes a sync job), `GET/DELETE /api/jobs/{id}`, `DELETE /api/jobs`, `POST /api/jobs/bulk-delete`, `PATCH /api/jobs/{id}/settings`, `POST …/cancel`, `POST …/translate` |
| Editing | `POST …/segments`, `DELETE …/segments/{id}`, `DELETE …/segments`, `PATCH …/segments/{id}`, `PATCH …/segments`, `POST …/segments/move`, `POST …/retime` |
| History | `GET …/history`, `POST …/history/undo`, `POST …/history/redo`, `POST …/history/{entry_id}/restore` |
| Translation tools | `PUT …/glossary`, `POST …/detect-speakers`, `GET …/qa`, `GET /api/glossary-profiles` |
| Export | `GET …/export/srt`, `…/export/vtt`, `…/export/log`, `POST …/export/save-next-to-video` (writes `Movie.he.srt` into the video's folder; `exists` answer instead of overwriting unless `overwrite`; refused for uploads) |
| Settings | `GET/PUT /api/settings`, `GET/PUT /api/settings/api-key`, `GET/POST /api/settings/keys`, `PUT/DELETE /api/settings/keys/{key_id}`, `POST /api/settings/providers/refresh`, `POST /api/settings/estimate`, `POST /api/settings/check`, `GET /api/settings/ollama/status`, `POST /api/settings/ollama/setup` |
| Storage | `GET /api/storage`, `POST /api/storage/clean` |
| Live | `WS /ws/jobs/{job_id}` |

---

## 7. Frontend

- **Structure:** `components/` → `ingest/` (media drop & season batch queue, stream selector with embedded subtitle track picker, mode switch *Create subtitles / Sync my subtitles*, `GenrePresets` smart workflow selector, `PipelineConfig` with section filter tabs, Smart Audio Processing toggle, collapsible Advanced Audio drawer (`vad_filter`, `high_speech_sensitivity`, `vocal_isolation_mode`: `off`/`auto`/`on`), `transcription_provider` (`faster_whisper`/`whisperx`), shared `initial_prompt` / context glossary input, and exclusive per-video formatting toggles `convert_to_metric`, `remove_sdh_speakers`, `skip_end_credits`, and `onscreen_position`, plus `SyncOptions`), `monitor/` (live pipeline bar with Smart Audio Map / Vocals Isolated badges, stream feed, token & local AI savings monitor), `studio/` (dual-layer video player, dual-lane `WaveformTimeline` with live drag HUD overlay, virtualized subtitle table/rows with select-all, fast range selection, quick jump, read-only *Speaking to* listener tags & categorized problem/tag filters — `SubtitleTable` composes the hooks and toolbar pieces in `studio/table/` (`useSegmentFilters`, `useRowSelection`, `useVirtualRows`, `useRowDragDrop`, `useQualityCheck`, `SearchReplaceBar`, `TagFilterSelect`, `JumpToRow`, `ProblemFilterSelect`, `TableActions`, `QaPanel`, `SelectionBar`) — live onscreen position selector, export toolbar, glossary & cast speakers / row-translate (`RowTranslateDialog` strictly filtered to active translation-capable models) / insert-line dialogs), `settings/` (`SettingsScreen` compact single-page Bento layout with section navigation tabs `all`/`keys`/`models`/`cost`/`system`: inline header with Appearance & Back, collapsible multi-provider `ApiKeySection` modal & tabbed Key Vault, 3-column per-stage AI `ModelCard` grid with Vision frame detail & max width, Flex & GPU/VRAM-aware Best-Value Auto-Selector, 2-column Cost-Saver + collapsible Ollama guide paired with collapsible Cost Estimator, and 2×3 `StorageSection` grid paired with Diagnostics & Feedback), `layout/` (Sidebar with active stage-provider summary badge, Studio, Settings, Help buttons, GPU status & bulk project deletion; Header), `HelpSection` (standalone view with 4-step workflow guide and interactive Q&A accordion).
- **State:** Zustand — `useJobStore` (jobs, active job, segments, WebSocket, ingest mode, sync file), `useSettingsStore`, `useUiStore` (language, theme, view: `'studio' | 'settings' | 'help'`; persisted to `localStorage`, reflected on `<html>` for CSS and RTL layout).
- **Contracts:** `types/index.ts` mirrors the backend schemas; `services/api.ts` is the only REST client.
- **i18n:** `i18n/en.ts` + `i18n/he.ts`; message keys are typed, so a missing translation fails the TypeScript build. Hebrew switches the whole layout to RTL (`dir`), with explicit `dir` only around genuinely left-to-right fragments.
- **Player:** HTML5 video with Range seeking and a dynamic WebVTT blob track, so edits show on the video immediately. Shortcuts: `Space`, `←/→` (±2 s, `Shift` ±0.1 s), `[` `]` (set start/end to playhead), `N` (add line), `F` (fullscreen).
- **Dev/prod:** `npm run dev` on `:5173` proxies `/api` and `/ws` to `:8000`; `npm run build` produces `frontend/dist`, which FastAPI serves on the same origin.
- **Desktop bridge:** `utils/desktop.ts` talks to `window.pywebview.api` when present:
  - `resolveDroppedFilePath` passes dropped `FileList` objects through WebView2's `chrome.webview.postMessageWithAdditionalObjects('FilesDropped', files)` and queries `window.pywebview.api.resolve_dropped_file(filename)` (`webview.dom._dnd_state`) to obtain the exact local disk path (`CoreWebView2File.Path` on Windows, `NSURL` on macOS, `QUrl` on Linux) without copying or uploading to temp; falls back to browser upload only in an external browser.
  - `pickLocalMediaFile` calls `window.pywebview.api.pick_media_file()` (`window.create_file_dialog`) to open the native OS file picker and return the local path directly.
  - `set_fullscreen` toggles fullscreen of the whole desktop window; `read_clipboard` backs the path field's right-click *Paste* menu (since WebView2 may refuse the page's clipboard permission — browsers fall back to `navigator.clipboard`).

---

## 8. Desktop shell and auto-updater (`packaging/`, `core/updater.py`)

- **`launcher.py`** starts Uvicorn inside the same process (server thread), then shows the studio in a **pywebview window** (Edge WebView2) with a **tray icon** (`pystray`): open, open in browser, data folder, `.env`, server log, restart, quit. A loading page shows while the server starts; window size is remembered; closing asks first if a project is still running (quit / keep in background / cancel).
- **Fallbacks:** `--browser` (or `SUBBUILD_UI=browser`), or automatically when WebView2 is missing; `--no-tray` for a console server. A second launch asks the running copy to show its window (`POST /api/desktop/show`); if the running copy has no window it says so instead of opening a tab.
- **Data folder** (`SUBBUILD_HOME` → `data` next to the app → `%LOCALAPPDATA%\SubBuild Studio`) holds `.env`, `storage/`, `launcher.log` and the WebView profile, so updating never touches user data. `run.bat` sets `SUBBUILD_HOME=backend`, keeping the developer layout.
- **Source execution & FFmpeg:** stdout/stderr go to `launcher.log`; child consoles (ffmpeg, nvidia-smi) are hidden under `pythonw`; `prepare_environment()` also adds `packaging/ffmpeg` to `PATH` when present. `run.bat` / `./run.sh --window` run the launcher from source using `packaging/requirements-desktop.txt`.
- **Auto-updater:** Root `VERSION` and `core/updater.py` compare the installed version/tags against published remote semver release tags (`vX.Y.Z`) only (ignoring raw branch commits). `run.bat` and `run.sh` invoke `--auto --no-frontend-build` before launch (skipping feature branches and dirty worktrees automatically; bypass with `--no-update` or `SUBBUILD_NO_UPDATE=1`). In-app Sidebar badges and Settings controls let users check and apply release updates from the UI (blocked with `409 Conflict` if any pipeline job is active).

---

## 9. Security and privacy
- Server binds to `127.0.0.1`; CORS allow-list (`cors_origins`); `TrustedHostMiddleware` (`ALLOWED_HOSTS`) blocks DNS rebinding.
- **CSRF guard** (`core/request_guard.py`): CORS only hides responses from other websites; it does not stop a browser from *sending*
  simple requests (a body-less POST, a form, a multipart upload) to `127.0.0.1`. Every state-changing request (not GET/HEAD/OPTIONS)
  and every WebSocket handshake must therefore carry an `Origin` that is in `cors_origins` or equal to the request's own host;
  `Origin: null` is rejected. Requests without `Origin` (desktop launcher, curl, tests) are allowed unless `Sec-Fetch-Site` marks them
  as cross-site. Rejected requests get `403` (WebSockets close with `1008`) and are logged.
- Changing `HOST` to `0.0.0.0` requires adding the LAN names to `ALLOWED_HOSTS` and exposes a file-reading API — documented as unsafe by default.
- API keys: keys saved in the Settings Multi-Provider Key Vault (or the first-run dialog) are persisted to `storage/provider_keys.json` (git-ignored inside `storage/`) and returned masked in API responses, while legacy Google keys via `PUT /api/settings/api-key` also write to `.env` (git-ignored);
  a key typed in the Ingest screen is used in memory for that run only and never written to disk. `.gitignore` also covers `*.exe`, `build/`, `dist/`, `storage/`, `data/` and `launcher.log`.
- Sent to AI providers: subtitle text (and glossary), sampled frames when on-screen detection/speaker detection is used, nothing else. Audio and video stay local; when using local Ollama models, text and frames stay 100% local on the machine.
- Housekeeping and cleanup only delete files inside `storage/` that the app created.

---

## 10. Testing and quality gates
- Backend: `pytest` in `backend/tests` (820 tests across `unit`, `api`, `pipeline`, `security`, and `smoke` markers). Whisper and AI providers (Google Gemini, OpenAI, Anthropic, Ollama) are replaced by fakes; ffmpeg-based tests run end to end on generated clips (forced track, frame scan, whole pipeline, sync with synthetic delays/frame-rate/cuts). Areas: RTL fixer, SRT formatter, refine quality, Smart Audio Map window analysis & selective conditioning, Demucs vocal isolation & WhisperX phoneme alignment/interjection filtering, long-film diarization cluster re-split & voice-pitch gender, speaker vs. addressee gender resolution, speaker split & SDH cleanup, smart end-credits detection & post-credits scene preservation (`credits_detect`), embedded subtitle extraction, missing-speech Whisper supplement, FinOps local pre-filtering & Ollama pre-tagging, Whisper provider (cancel, fallback), multi-provider LLM translation/discovery/key-vault & Google Flex temporary cooldown/re-probe, GPU/VRAM-aware best-value auto-recommendation, settings/pricing, foreign speech, gap recovery, word snap, retime, row translate, glossary profiles, storage cleanup, sync, insert line, Git/Release auto-updater, desktop launcher.
- Frontend: `npx tsc -b`, `npx oxlint src` (0 errors), and `vitest run` (85 tests across 16 spec files in `frontend/src/__tests__/`), plus `vite build`; UI & API flows are checked black-box by the Java 21 QA suite (`53` regression tests: `35` RestAssured API/Security + `18` Selenium WebDriver E2E in Chrome across English and Hebrew, including `9` smoke tests).
- Windows-only behavior (tray, WebView2 window, CUDA loading) is verified manually on a Windows PC; the launcher's pure logic is unit-tested.

---

## 11. Extending the system
- **New transcription or translation engine:** implement `BaseTranscriber` / `BaseTranslator` and return the shared dataclasses (`RawSegment`, `TranslatableSegment`). Each stage builds its provider itself (`TranscribeStage`, `TranslateStage._make_translator`) and also accepts an injected one, which is how the tests use fakes; multi-provider LLM/Vision calls route through `providers/llm/client.py`.
- **New pipeline stage:** subclass `PipelineStage` (`pipeline/stages/base.py`: a `name` and `async execute(ctx, on_progress)`), add a `JobStatus` value if it deserves its own progress state, add it to `PipelineOrchestrator._stages()`, add the matching label in the frontend `LivePipelineBar` and both i18n files.
- **New AI provider model or price change:** edit `core/provider_catalog.py` (pricing, capabilities, task eligibility, thinking/reasoning modes, and live discovery filtering/dedup rules) or `core/model_catalog.py` for Gemini historical promotions.
- **New DB column:** add it to the model and to `_ADDED_COLUMNS` in `core/database.py`.
- **New UI language:** add `i18n/<code>.ts` with the same typed keys and extend `Language` in `useUiStore`.
