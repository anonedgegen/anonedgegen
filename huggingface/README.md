# HuggingFace Dataset Scripts

Scripts for uploading and downloading the [anonedgegen/anonedgegen](https://huggingface.co/datasets/anonedgegen/anonedgegen) dataset.

- **Upload** — requires a HuggingFace write token (authorized users only)
- **Download** — public, no token required

---

## Installation

**1. Create and activate a virtual environment**

```bash
python -m venv .hfvenv
source .hfvenv/bin/activate        # macOS / Linux
# .hfvenv\Scripts\activate         # Windows
```

**2. Install dependencies**

```bash
pip install -r huggingface/requirements_hf.txt
```

---

## Upload

Pushes a local folder to the dataset repository. A valid HuggingFace write token is required.

Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — select **Write** access.

**Set your token as an environment variable (recommended):**

```bash
export HF_TOKEN=hf_your_token_here
```

**Run the upload:**

```bash
# Upload the current directory
python huggingface/upload_dataset.py .

# Upload a specific folder
python huggingface/upload_dataset.py /path/to/your/data

# Upload into a specific folder inside the repo
python huggingface/upload_dataset.py /path/to/your/data --folder my_folder

# With a custom commit message
python huggingface/upload_dataset.py /path/to/your/data --message "Add training split v2"

# Pass the token directly (alternative to env var)
python huggingface/upload_dataset.py . --token hf_your_token_here
```

If your token is missing, expired, or lacks write access, the script will exit immediately with a clear error before uploading anything.

---

## Download

Downloads the dataset files to a local directory exactly as they were uploaded. No token or account required.

```bash
# Download all files to ./anonedgegen_dataset (default)
python huggingface/download_dataset.py

# Specify an output directory
python huggingface/download_dataset.py --output-dir ./my_data

# Download only a specific folder
python huggingface/download_dataset.py --folder my_folder

# Use a custom HuggingFace cache directory
python huggingface/download_dataset.py --cache-dir /tmp/hf_cache
```

Files are downloaded as-is (e.g. `test.json` stays `test.json`), mirroring what was uploaded with `upload_dataset.py`.

---

## Arguments Reference

### `upload_dataset.py`

| Argument | Type | Default | Description |
|---|---|---|---|
| `folder_path` | positional | `.` | Local folder to upload |
| `--token` | optional | `$HF_TOKEN` | HuggingFace write token |
| `--folder` | optional | repo root | Destination folder inside the repo |
| `--message` | optional | `"Upload dataset"` | Commit message |

### `download_dataset.py`

| Argument | Type | Default | Description |
|---|---|---|---|
| `--output-dir` | optional | `./anonedgegen_dataset` | Local save path |
| `--folder` | optional | all files | Download only a specific folder |
| `--cache-dir` | optional | HF default | Custom HuggingFace cache path |
