# simple-tts

A simple CLI for converting documents to speech using [Kokoro TTS](https://github.com/hexgrad/kokoro).

## Features

- **Multi-format support**: PDF, DOCX, EPUB, Markdown, TXT, HTML, and more
- **50+ voices**: American/British English, Spanish, French, Italian, Japanese, Korean, Chinese, Portuguese, Hindi
- **Multiple output formats**: M4A, MP3, WAV, OGG
- **Captions by default**: Writes a WebVTT sidecar for easy navigation (use `--no-vtt` to skip)
- **Adjustable speed**: 0.5x to 2.0x playback speed
- **Progress tracking**: Real-time progress with time estimates

## Installation

### System Dependencies

```bash
# macOS
brew install poppler pandoc ffmpeg

# Ubuntu/Debian
sudo apt install poppler-utils pandoc ffmpeg
```

### Install Package

```bash
pip install simple-tts
```

Or from source:

```bash
git clone https://github.com/maehr/simple-tts
cd simple-tts
pip install -e .
```

## Usage

### Convert a Document

```bash
# Basic usage with default voice (af_bella)
simple-tts convert document.pdf

# Specify voice and output file
simple-tts convert document.pdf --voice bf_emma --output audio.mp3

# Faster speed, skip confirmation
simple-tts convert essay.md -v am_adam -s 1.3 -y

# Convert the bundled EPUB sample
simple-tts convert tests/minimal.epub -y

# Skip captions if you don't need them
simple-tts convert document.pdf --no-vtt
```

### Example Output

```
$ simple-tts convert README.md --voice af_bella -y

📄 Input: /path/to/README.md
🔊 Output: /path/to/README.m4a
🗣️ Voice: af_bella
⠋ Extracting text...     ━━━━━━━━━━━━━━━━ 100% 0:00:00 • -:--:--
  └─ 2,847 chars in 0s
⠋ Preparing text...      ━━━━━━━━━━━━━━━━ 100% 0:00:00 • -:--:--
  └─ 3 chunks in 0s
⠋ Loading TTS model...   ━━━━━━━━━━━━━━━━ 100% 0:00:02 • -:--:--
  └─ Model loaded in 2s
⠋ Synthesizing (3/3)...  ━━━━━━━━━━━━━━━━ 100% 0:00:15 • -:--:--
  └─ 3 chunks in 15s
⠋ Saving audio...        ━━━━━━━━━━━━━━━━ 100% 0:00:00 • -:--:--
  └─ Saved in 0s

           Summary
┌──────────────────┬──────────────┐
│ Output File      │ README.m4a   │
│ Audio Duration   │ 2m 34s       │
│ Processing Time  │ 18s          │
│ Speed Ratio      │ 8.6x realtime│
└──────────────────┴──────────────┘

✅ Done!
```

### List Available Voices

```bash
simple-tts voices
```

Output:

```
                Available Voices
┌────────────────┬─────────────────┬──────────────────────────┐
│ Voice ID       │ Language        │ Description              │
├────────────────┼─────────────────┼──────────────────────────┤
│ af_bella       │ English (US)    │ American female voice    │
│ af_nicole      │ English (US)    │ American female voice    │
│ am_adam        │ English (US)    │ American male voice      │
│ bf_emma        │ English (UK)    │ British female voice     │
│ bm_george      │ English (UK)    │ British male voice       │
│ ef_dora        │ Spanish         │ Spanish female voice     │
│ ff_siwis       │ French          │ French female voice      │
│ if_sara        │ Italian         │ Italian female voice     │
│ jf_alpha       │ Japanese        │ Japanese female voice    │
│ kf_alpha       │ Korean          │ Korean female voice      │
│ pf_dora        │ Portuguese (BR) │ Brazilian female voice   │
│ zf_xiaobei     │ Chinese         │ Chinese female voice     │
│ ...            │ ...             │ ...                      │
└────────────────┴─────────────────┴──────────────────────────┘
```

### CLI Options

```bash
simple-tts convert --help
```

```
Usage: simple-tts convert [OPTIONS] INPUT_FILE

  Convert a document to speech using Kokoro TTS.

Arguments:
  INPUT_FILE  Input file to convert (PDF, DOCX, EPUB, MD, TXT, etc.)  [required]

Options:
  -v, --voice TEXT    Voice ID (run 'simple-tts voices' to list)  [default: af_bella]
  -o, --output PATH   Output file path
  -s, --speed FLOAT   Playback speed (0.5-2.0)  [default: 1.1]
  -f, --format TEXT   Output format (m4a, mp3, wav, ogg)  [default: m4a]
  -y, --yes           Skip confirmation prompt
  --help              Show this message and exit.
```

### Check Dependencies

```bash
simple-tts check-deps
```

```
System Dependencies Status:

  ✓ pdftotext - PDF text extraction (from Poppler)
  ✓ pandoc - Universal document converter
  ✓ ffmpeg - Audio encoding and processing
```

## Supported Languages

| Language   | Voices                              |
|------------|-------------------------------------|
| English US | af_bella, af_nicole, am_adam, etc.  |
| English UK | bf_emma, bf_isabella, bm_george     |
| Spanish    | ef_dora, em_alex                    |
| French     | ff_siwis                            |
| Italian    | if_sara, im_nicola                  |
| Japanese   | jf_alpha, jf_gongitsune, jm_kumo    |
| Korean     | kf_alpha, km_alpha                  |
| Portuguese | pf_dora, pm_alex                    |
| Chinese    | zf_xiaobei, zm_yunxi                |
| Hindi      | hf_alpha, hm_omega                  |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check src tests
ruff format src tests

# Run tests
pytest

# Type checking
ty check src
```

## License

AGPL-3.0 - See [LICENSE](LICENSE) for details.
