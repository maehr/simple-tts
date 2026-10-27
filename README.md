# simple-tts

Convert prepared text and structured documents to speech with [Phonon 40M Multilingual](https://huggingface.co/gradium/phonon-40m-multilingual).
simple-tts reads English, German, Spanish, and French text.
Each language uses voices from the [Phonon voice catalogue](https://huggingface.co/gradium/phonon-voices).

**License restriction:**
The Phonon 40M Multilingual weights use the CC BY-NC 4.0 license.
This license does not permit commercial use.
The model is gated on Hugging Face.
simple-tts does not include the weights.
See [License](#license).

## Features

- Read Markdown, QMD, RMD, TXT, DOCX, EPUB, RST, and HTML files.
- Narrate English, German, Spanish, or French text. Each conversion uses one language.
- Select catalogue voices by name or by voice ID. Each language has a default voice.
- Rotate voices of the same language across chapters.
- Change the speaking rate with `--duration-delta`.
- Write M4A, M4B, MP4, MP3, WAV, or Opus audio.
- Add embedded chapters to M4B and MP4 files.
- Run on the CPU with 8-bit weights by default. Select Metal or CUDA explicitly.
- Keep audio memory use bounded by the text chunk size.
- Run a local MCP server with cancellable conversion jobs.

## Installation

Install the system tools for your platform.

```bash
# macOS
brew install pandoc ffmpeg

# Ubuntu or Debian
sudo apt install pandoc ffmpeg
```

Install the package with Python 3.12 or later.

```bash
pip install simple-tts
```

To install from source, use these commands.

```bash
git clone https://github.com/maehr/simple-tts
cd simple-tts
pip install -e .
```

The project uses Python 3.12 for development.
The package installs the Rust-based `ptts` runtime, version 0.4.2.
It does not install PyTorch.
The FFmpeg build must include the encoder for the selected format.

### Get access to the model

The model `gradium/phonon-40m-multilingual` is gated.
You must accept its terms before the first conversion.

1. Sign in to [Hugging Face](https://huggingface.co).
2. Open the [model page](https://huggingface.co/gradium/phonon-40m-multilingual).
3. Read the CC BY-NC 4.0 license.
4. Accept the terms of the model.
5. Install the Hugging Face CLI from the separate `huggingface_hub` package.
6. Log in with `hf auth login`.

```bash
pip install -U huggingface_hub
hf auth login
```

As an alternative to the login, set the `HF_TOKEN` environment variable to a Hugging Face access token.
Keep the token secret.
Do not write the token into a file in a repository.

### Model and voice files

simple-tts downloads the model, the voice catalogue, and the voices from Hugging Face.
It uses these pinned revisions:

| Item | Revision |
| --- | --- |
| `ptts` runtime | `0.4.2` |
| Model `gradium/phonon-40m-multilingual` | `1537ce46c64f42903e0d608b132ccb073dac46d2` |
| Voice catalogue `gradium/phonon-voices` | `da6c9ec0d47be8df5c87c730a3aa97187f3229c3` (the `ptts` default) |

Each file downloads one time, at its first use:

| File | Size | First use |
| --- | --- | --- |
| Voice catalogue `voices.jsonl` | 33 KB | The first `voices` command, `list_voices` call, or conversion. |
| 8-bit weights `model.q8.gguf` | 74.6 MB | The first conversion with `--device auto` or `cpu`. |
| f32 weights `model.safetensors` | 121.7 MB | The first conversion with `--device mps` or `cuda`. |
| One voice | 256 KB | The first chapter that uses the voice. |

The first model download also adds `config.json`, `tokenizer.json`, and `default-voice.safetensors`.

The files go into the Hugging Face cache.
simple-tts finds the cache in this order, as `ptts` does:

1. `$HF_HUB_CACHE`, or the older `$HUGGINGFACE_HUB_CACHE`
2. `$HF_HOME/hub`
3. `$XDG_CACHE_HOME/huggingface/hub`
4. `~/.cache/huggingface/hub`

### Use the cache without a network

simple-tts loads a complete cached snapshot of the pinned model revision as a local folder.
A complete snapshot holds `config.json`, `tokenizer.json`, and the weight file for the device.
If the snapshot does not have all files, simple-tts loads `hf://gradium/phonon-40m-multilingual@1537ce46c64f42903e0d608b132ccb073dac46d2`.
This source downloads the missing files.

Local tests with `ptts` 0.4.0 and 0.4.2 gave these results:

- A second process loaded the cached snapshot and sent no request to the Hub.
- After the first download, `ptts` read the cached voice catalogue and sent no request.
- A cached voice loaded without a request. A voice that is not in the cache needs a network connection.
- `ptts` ignored `HF_HUB_OFFLINE=1`. This variable did not stop a download.

To prepare for work without a network, do these steps while you are online:

1. Run `simple-tts voices` to cache the voice catalogue.
2. Convert a short text with each voice that you plan to use.
3. If you plan to use Metal, repeat step 2 with `--device mps`.

If a voice is not in the cache and the network is not available, the conversion stops at the first chapter of that voice.
The command then writes no output file.

### Use a local checkpoint

Set `SIMPLE_TTS_MODEL` to load a different checkpoint.
The value can be a local checkpoint folder.
Use a copy of a complete model snapshot from the Hugging Face cache.
The folder must contain `config.json`, `tokenizer.json`, and the weight file for the device.
A test load from a local snapshot folder sent no request to the Hub.

```bash
export SIMPLE_TTS_MODEL=~/models/phonon-40m-multilingual
simple-tts convert paper.md
```

The value can also be a pinned Hub source, such as `hf://gradium/phonon-40m-multilingual@<revision>`.
A Hub source sends one request to the Hub on every load, also when the files are in the cache.
Use a local folder for work without a network.

## Usage

### Prepare input

Input must contain readable prose in the correct order.
Check the reading order before conversion.
Remove unwanted text, such as page numbers, repeated headers, and footnotes.
Markdown headings define chapters.

Use one language for each document.
simple-tts does not detect the language and does not translate.
Select the language with `--language`.

Save this example as `paper.md`:

```markdown
# Introduction

The paper examines the history of programming languages.

# Findings

The authors identify three main changes.
```

```bash
simple-tts convert paper.md --format m4b
```

Check numbers and years in the text.
The Phonon text normalizer can read a year as a cardinal number.
For example, it writes `1892` as "1 thousand 892" in English.
Write a year in words if its spoken form is important.

PDF input is not supported.
Convert PDFs with an external tool before you use simple-tts:

- [pdf-inspector](https://github.com/firecrawl/pdf-inspector): PDF-to-Markdown conversion.
- [Poppler pdftotext](https://manpages.debian.org/unstable/poppler-utils/pdftotext.1.en.html): plain-text extraction.
- [OCRmyPDF](https://ocrmypdf.readthedocs.io/en/latest/cookbook.html): text recognition for scans.

These tools are optional and separate from simple-tts.
Check the extracted text before conversion.

### Convert a document

M4A is the default format.
English (`en`) is the default language.
Each language has a default voice.

```bash
simple-tts convert paper.md
```

An explicit output suffix selects the format.

```bash
simple-tts convert paper.md --voice Hadley --output audio.mp3
```

Use `--yes` to skip the confirmation prompt.

```bash
simple-tts convert essay.md --language en --voice Zoey -y
```

### Select a language and a voice

Use `--language` or `-l` to select the narration language.
Use `--voice` or `-v` to select a voice of that language.

| Code | Language | Default voice | Catalogue voices |
| --- | --- | --- | --- |
| `en` | English | Zoey | 37 |
| `de` | German | Nele | 13 |
| `es` | Spanish | Noa | 25 |
| `fr` | French | Noémie | 15 |

The voice counts apply to the pinned voice catalogue.

```bash
simple-tts convert paper.md --language en --voice Zoey
simple-tts convert aufsatz.md --language de --voice Nele
simple-tts convert articulo.md --language es --voice Noa
simple-tts convert article.md --language fr --voice Noémie
```

The CLI ignores the case of the language code.
A voice name matches without case or accents. For example, `noemie` selects Noémie.
A voice ID must match exactly. For example, `-l fr -v FXxJ9mANRq6BCTX5` selects Noémie.

The voice must speak the narration language.
simple-tts never replaces a voice with a voice of a different language.
For example, `-l de -v Zoey` fails before extraction.
The error names both languages and the command that lists the German voices.

The catalogue also contains Portuguese voices.
simple-tts does not support Portuguese in this release.

List the voices with these commands.

```bash
simple-tts voices
simple-tts voices --language de
```

The list shows the name, the voice ID, the language, and the catalogue description of each voice.
The text `(default)` marks the default voice of each language.
The catalogue descriptions call Zoey and Noémie playful and Noa bubbly.
Read the descriptions to select a different narrator for long texts.

### Rotate narrator voices

Separate voice names or IDs with commas to rotate narrators across chapters.

```bash
simple-tts convert buch.md --language de --voice Nele,Anton --format mp4
```

The first section uses the first voice.
An introduction counts as a section.
Each later section uses the next voice, then the rotation repeats.
Every chunk within a section uses the same voice.
All voices must speak the narration language.
The command rejects unknown voices, empty list entries, and voices of a different language before extraction.
simple-tts loads the model one time for all voices of a conversion.

### Change the speaking rate

Use `--duration-delta` to change the speaking rate.
A negative value speaks faster and makes the audio shorter.
A positive value speaks slower and makes the audio longer.
The value is a condition of the Phonon model.
It is not a speed multiplier.

The valid range is from -0.3 to 0.3.
The default is 0.0.
The CLI rejects values outside this range.

```bash
simple-tts convert article.md --language fr --duration-delta -0.2
```

The effect is not linear.
A local test measured these audio durations, relative to 0.0.
It used one passage of about 600 characters in each language, CPU with 8-bit weights, and the median of three seeds.

| `--duration-delta` | English | German | Spanish | French |
| --- | --- | --- | --- | --- |
| -0.3 | 0.83 | 0.84 | 0.86 | 0.85 |
| -0.1 | 0.93 | 0.92 | 0.94 | 0.93 |
| 0.1 | 1.06 | 1.09 | 1.08 | 1.04 |
| 0.3 | 1.12 | 1.17 | 1.15 | 1.10 |

At -0.3, the audio was 14 to 17 % shorter.
At 0.3, the audio was 10 to 17 % longer.
A listening check of these values is still pending.

The options `--speed` and `-s` are removed.
A `--speed` value has no exact `--duration-delta` equivalent.
Remove `--speed` from your scripts.
Then select a `--duration-delta` value.

### Chapters and navigation

Use `--format m4b` or `--format mp4` for AAC audio with embedded chapters.
M4A, M4B, and MP4 use a default bitrate of 128 kbps.

```bash
simple-tts convert book.md --format mp4 --chapter-level 2
```

This writes one file, `book.mp4`, with embedded chapter markers.
The command does not create separate chapter files or VTT files.

Only headings at the selected level start chapters (default level 1).
Valid chapter levels range from 1 through 6.

Each heading appears once in the narration, including headings at other levels.
Text before the first selected heading forms an `Introduction` chapter.
If no headings match, the chapter uses the input stem as its title.
All input formats use one Pandoc call before chapter extraction.

```bash
simple-tts convert book.qmd --output book.m4a --chapter-level 2
simple-tts convert book.rmd --output book.m4b --chapter-level 2
simple-tts convert tests/minimal.epub --format m4b
```

### Select a device

Use `--device` to select the hardware and the weight format.

| `--device` | Hardware | Weights |
| --- | --- | --- |
| `auto` (default) | CPU | 8-bit (`q8`) |
| `cpu` | CPU | 8-bit (`q8`) |
| `mps` | Apple GPU (Metal) | 32-bit float (`f32`) |
| `cuda` | NVIDIA GPU (CUDA) | 32-bit float (`f32`) |

`auto` always selects the CPU, also when Metal or CUDA is available.
8-bit weights run only on the CPU.
For this reason, `mps` and `cuda` use f32 weights.

```bash
simple-tts convert book.md --device mps
simple-tts convert book.md --device cuda
simple-tts convert book.md --device cpu
```

The command reports the model source, the device, and the weight format.
An explicit GPU device fails if the `ptts` build does not include it.
In this case, simple-tts does not use the CPU instead.
The `ptts` wheel for macOS on Apple silicon includes Metal and the CPU, but not CUDA.
CUDA is not tested.
simple-tts uses at most six CPU threads.

#### Why the CPU is the default

The developer of Phonon reports 20x real time on one MacBook Air M5 CPU core and 39x on four cores.
These numbers are vendor-reported.
They are not measurements of the multilingual model with 8-bit weights.
See the [Phonon performance notes](https://github.com/gradium-ai/xn-ptts#performance).

A local test on one Apple M5 Pro gave these results with `ptts` 0.4.0.
`ptts` 0.4.2 gave the same speed.
Each run converted a three-chapter document of about 3 KB in each language, from model load to the WAV file.

| Configuration | Speed (real time) | Peak memory | GPU use |
| --- | --- | --- | --- |
| CPU, 8-bit, 6 threads (simple-tts) | 70.1x to 71.3x | 393 MB | 0 % |
| Metal, f32 | 31.4x to 31.6x | 477 MB, plus about 300 MB of GPU memory | 82 % (median) |

On this machine, Metal with f32 weights was slower than the CPU with 8-bit weights.
The 8-bit weight file is smaller: 74.6 MB, compared to 121.7 MB for f32.
The CPU setting also keeps the GPU free for other work.
These results apply only to the tested machine.

8-bit weights can change the speech.
A listening comparison of 8-bit and f32 output is still pending.
If you hear a problem with the default, compare the output of `--device mps`.

### Audio storage

The command writes each completed chunk to a temporary mono float32 PCM file at 24,000 Hz.
It encodes the audio once after synthesis completes.
Audio memory use stays within one text chunk.
It replaces the output file only after encoding succeeds.
Temporary PCM storage uses approximately 3.5 GB per ten hours of audio.

### CLI options

```bash
simple-tts convert --help
```

| Option | Purpose | Default |
| --- | --- | --- |
| `-l`, `--language` | Select the narration language: `en`, `de`, `es`, or `fr`. | `en` |
| `-v`, `--voice` | Select catalogue voice names or IDs, separated by commas, for chapter rotation. | The default voice of the language |
| `-o`, `--output` | Set the output path and infer its format. | Beside the input |
| `--duration-delta` | Set the speaking rate from -0.3 to 0.3. Negative values speak faster. | `0.0` |
| `-f`, `--format` | Select `m4a`, `m4b`, `mp4`, `mp3`, `wav`, or `opus`. | `m4a` |
| `--chapter-level` | Select the heading level from 1 through 6. | `1` |
| `--device` | Select `auto`, `cpu`, `mps`, or `cuda`. | `auto` |
| `-y`, `--yes` | Skip the confirmation prompt. | Off |

List the voices of one language with this command.

```bash
simple-tts voices --language fr
```

Check the system tools with this command.

```bash
simple-tts check-deps
```

### Troubleshooting

| Message | Cause and action |
| --- | --- |
| `No access to the gated model gradium/phonon-40m-multilingual.` | You did not accept the model terms, or you are not logged in. Do the steps in [Get access to the model](#get-access-to-the-model). |
| `Phonon cannot read the voice catalogue: ...` | The catalogue is not in the cache. Connect to the network and run `simple-tts voices`. |
| `Phonon cannot download a required file: ...` | A model file or a voice is not in the cache. Connect to the network and run the conversion again. |
| `Phonon cannot find a checkpoint or voice: ...` | Make sure that `SIMPLE_TTS_MODEL` points to a complete checkpoint folder. |
| `Unknown voice '...'. List voices with: simple-tts voices --language de` | The catalogue does not contain this name or voice ID. Run the command in the message and select a listed voice. |
| `Voice 'Zoey' is English (en), but the narration language is German (de).` | Select a voice of the narration language, or change `--language`. |
| `Requested device 'mps' is unavailable in this ptts build.` | The installed `ptts` build does not support this GPU. Use `--device auto`. |
| `No such option: --speed` | Use `--duration-delta`. See [Change the speaking rate](#change-the-speaking-rate). |

## Local MCP server

The optional Model Context Protocol (MCP) server supports local desktop assistants and coding agents.
The MCP server supports macOS and Linux.
On Windows, the MCP command exits with a platform error.
The `convert` command remains available on Windows.

Install the MCP extra with this command.

```bash
pip install "simple-tts[mcp]"
```

Start the server with an existing workspace directory.

```bash
simple-tts mcp --workspace /absolute/path/to/workspace
```

The server uses stdio.
Diagnostics use stderr.
The server reserves stdout for MCP messages.
The server does not read the voice catalogue or load the model at startup.
The first `list_voices` call or conversion can download the voice catalogue.
The first conversion can download the model and its voices.
[Get access to the model](#get-access-to-the-model) before the first conversion.

### Configure a client

Add this entry to your client's MCP configuration.
Replace both paths with absolute paths on your computer.

```json
{
  "mcpServers": {
    "simple-tts": {
      "command": "/absolute/path/to/venv/bin/simple-tts",
      "args": ["mcp", "--workspace", "/absolute/path/to/workspace"]
    }
  }
}
```

The executable must use an environment that contains `simple-tts[mcp]`.
Pandoc and FFmpeg must be available in the server's `PATH`.

### Tools

The server exposes five tools.

| Tool | Purpose |
| --- | --- |
| `list_voices(language)` | Return the narration languages and the catalogue voices. |
| `check_dependencies()` | Return Pandoc and FFmpeg availability. |
| `start_conversion(...)` | Validate a request and return a job ID. |
| `get_conversion(job_id)` | Return progress, errors, or completed audio metadata. |
| `cancel_conversion(job_id)` | Request cancellation and return the current job state. |

### List voices

`list_voices` reads only the voice catalogue.
It loads no model weights and no voices.
The optional `language` argument accepts `en`, `de`, `es`, or `fr`.
It limits the `voices` list to one language.
The `languages` list always contains all four languages.
The result does not contain Portuguese voices.

```json
{
  "languages": [
    {"code": "en", "name": "English", "default_voice": "Zoey"},
    {"code": "de", "name": "German", "default_voice": "Nele"},
    {"code": "es", "name": "Spanish", "default_voice": "Noa"},
    {"code": "fr", "name": "French", "default_voice": "Noémie"}
  ],
  "voices": [
    {
      "name": "Nele",
      "id": "2A2fFf8SkDEJ8nzx",
      "language": "de",
      "description": "<catalogue description>",
      "default": true
    }
  ]
}
```

If the catalogue is not in the cache and the network is not available, `list_voices` returns a tool error.

### Convert text or a document

Call `start_conversion` with exactly one source: `text` or `input_file`.
The server reads supplied text as Markdown through Pandoc.
Use these tool arguments for German text with two rotating voices.

```json
{
  "text": "# Einleitung\n\nDas ist vorbereiteter Text.\n\n# Ergebnisse\n\nDas sind die Ergebnisse.",
  "language": "de",
  "voices": ["Nele", "Anton"],
  "format": "m4b"
}
```

Use these tool arguments for a French document within the workspace.

```json
{
  "input_file": "livre.md",
  "output": "livre.m4b",
  "language": "fr",
  "voices": ["Noémie"],
  "duration_delta": -0.1,
  "chapter_level": 1,
  "device": "auto"
}
```

| Argument | Value | Default |
| --- | --- | --- |
| `input_file` | A document path in the workspace. | None |
| `text` | Markdown text. | None |
| `language` | `en`, `de`, `es`, or `fr`, in lower case. | `"en"` |
| `voices` | A list of voice names or voice IDs. An empty list selects the default voice of the language. | `[]` |
| `duration_delta` | A number from -0.3 to 0.3. | `0.0` |
| `format` | `m4a`, `m4b`, `mp4`, `mp3`, `wav`, or `opus`. | None |
| `output` | An output path in the workspace. | None |
| `chapter_level` | An integer from 1 through 6. | `1` |
| `device` | `auto`, `cpu`, `mps`, or `cuda`. | `"auto"` |

The server applies the CLI rules for languages, voices, chapters, the speaking rate, formats, and devices.
`start_conversion` has no `speed` argument.
The server rejects a request that contains `speed`.
An output suffix selects the format.
The format defaults to `m4a` when neither option selects it.

Paths are relative to the workspace.
Absolute paths must remain within the workspace.
The server rejects paths and symlinks that escape the workspace.

MCP extraction uses Pandoc's `--sandbox` mode.
Include directives and embedded frames cannot add secondary files to narration.
Prepare complete narration before conversion.
Pandoc must support `--sandbox`.
If a reader requires data files, use a Pandoc build with embedded data files.
See the [Pandoc sandbox documentation](https://pandoc.org/MANUAL.html#option--sandbox).

Custom outputs require existing parent directories.
The server rejects existing outputs, including files created during conversion.
Default outputs use `.simple-tts/audio/<job_id>.<format>` within the workspace.

### Poll or cancel a job

`start_conversion` returns a `job_id` before extraction starts.
The server permits one active conversion at a time.
Another start request returns a busy error.

Call `get_conversion` with the returned ID.

```json
{"job_id": "<job_id>"}
```

The response includes `state`, `phase`, `completed_chunks`, and `total_chunks`.
The total chunk count is `null` until extraction completes.
States are `running`, `cancelling`, `succeeded`, `failed`, and `cancelled`.

The server checks the request syntax before it starts a job.
Syntax errors are MCP tool errors.
Examples are a missing source, an unsupported language, an empty voice name, a `duration_delta` outside the range, and an unknown argument such as `speed`.
The job worker checks the voices against the catalogue.
An unknown voice or a voice of a different language makes the job fail.
The job then has the state `failed`, and its `error` field gives the reason.
Model, device, and download errors also appear in the `error` field.

A successful job includes a `result` object.
It contains `path`, `uri`, `format`, `duration`, `processing_time`, and `chapters`.
Durations and chapter boundaries use seconds.
Each chapter contains `title`, `start`, and `end`.
File URIs require access to the server's local filesystem.
The server does not send audio bytes through MCP.

Call `cancel_conversion` with the same job ID to request cancellation.
The worker checks cancellation between stages, chunks, and audio frames.
After five seconds, the server terminates remaining work and external tools.
The server checks cancellation before it publishes the final file.

Job records remain in memory until shutdown.
Shutdown cancels active work and removes temporary files.
Completed audio files remain in the workspace.
Jobs do not resume after a server restart.

## Quality evidence

The developer of Phonon published a blind listening comparison with Kokoro.
The table shows these vendor-reported Elo scores from the [model card](https://huggingface.co/gradium/phonon-40m-multilingual), revision `1537ce46`.

| Language | Phonon 40M Multilingual | Kokoro |
| --- | --- | --- |
| English | 2097 | 2043 |
| French | 2016 | 1578 |
| Spanish | 2076 | 1629 |
| German | 2046 | No native German baseline |

The developer did these evaluations.
They are not independent tests.
They do not show that every Phonon voice is better for long narration.

A human listening review is still pending.
It covers a native voice in each language, 8-bit compared to f32 weights, and Kokoro in English, French, and Spanish.
The 0.4.0 release requires this review.

[Parler-TTS Mini Multilingual v1.1](https://huggingface.co/parler-tts/parler-tts-mini-multilingual-v1.1) is a possible alternative.
Its weights use the Apache 2.0 license.
It has about 0.9 billion parameters and about 3.75 GB of f32 weights.
simple-tts does not implement Parler-TTS.
The project considers it only if the license or the voice quality rules out Phonon.

## Development

Install the locked development dependencies.

```bash
uv sync --locked --group dev --extra mcp
```

Run the tests and code checks.

```bash
uv run --locked --group dev --extra mcp pytest tests src/simple_tts
uv run --locked --group dev --extra mcp ruff check src tests
uv run --locked --group dev --extra mcp ruff format --check src tests
uv run --locked --group dev --extra mcp ty check src
```

Unit tests replace `ptts` with a fake module.
They download no model weights, no voices, and no voice catalogue.
They use real Pandoc, FFmpeg, and ffprobe.
MCP tests use FastMCP's client and spawned workers with the fake module.
CI also checks the CLI without the MCP extra.

Tests with the real model are optional.
They need access to the gated model and the variable `SIMPLE_TTS_MODEL_TESTS=1`.

```bash
SIMPLE_TTS_MODEL_TESTS=1 uv run --locked --group dev --extra mcp pytest -m model
```

Never commit model weights, voice files, cache files, or generated audio.

## Versioning

The project follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Releases before `1.0.0` have an unstable public interface.
The public interface includes CLI options, output formats, and documented Python helpers.

Before `1.0.0`, the project uses these rules:

- Compatible fixes increase the patch version: `0.2.0` to `0.2.1`.
- New features or breaking changes increase the minor version: `0.2.0` to `0.3.0`.
- A minor increase resets the patch version to zero.
- Breaking changes include a `BREAKING CHANGE:` note in the changelog.

Version `1.0.0` will mark a stable public interface.
See [CHANGELOG.md](CHANGELOG.md) for changes in each version.

## License

simple-tts uses the AGPL-3.0 license. See [LICENSE](LICENSE) for the terms.

The speech components have separate licenses:

| Component | License | Access |
| --- | --- | --- |
| [`ptts` runtime](https://github.com/gradium-ai/xn-ptts) | MIT OR Apache-2.0 | Installed from PyPI |
| [Phonon 40M Multilingual](https://huggingface.co/gradium/phonon-40m-multilingual) model | CC BY-NC 4.0, no commercial use | Gated on Hugging Face |
| [Phonon voice catalogue](https://huggingface.co/gradium/phonon-voices) | Its own terms. The card metadata states `apache-2.0`. | Not gated |

The model card states: "The weights, tokenizer and voices in this repository are released under CC BY-NC 4.0".
A permissive runtime license does not remove this restriction.
Use the model only for a noncommercial purpose.
Read the terms of the voice catalogue before you use its voices.

simple-tts does not include or distribute model weights, voices, or the voice catalogue.
Each user downloads these files from Hugging Face and accepts their terms.
The license of simple-tts does not cover these files.
