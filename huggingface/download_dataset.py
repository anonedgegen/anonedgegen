import argparse
import pathlib
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

TARGET_REPO = "anonedgegen/anonedgegen"
DEFAULT_OUTPUT_DIR = "./anonedgegen_dataset"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the anonedgegen dataset from HuggingFace (no token required)."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local directory to save the dataset (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Custom HuggingFace cache directory.",
    )
    parser.add_argument(
        "--folder",
        default=None,
        help="Download only a specific folder, e.g. --folder my_folder",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {TARGET_REPO}")
    if args.folder:
        print(f"  Folder filter: {args.folder}")
    print(f"Saving to: {output.resolve()}")

    allow_patterns = f"{args.folder}/*" if args.folder else None

    try:
        snapshot_download(
            repo_id=TARGET_REPO,
            repo_type="dataset",
            local_dir=str(output),
            cache_dir=args.cache_dir,
            allow_patterns=allow_patterns,
        )
    except RepositoryNotFoundError:
        sys.exit(
            f"Error: Dataset '{TARGET_REPO}' not found on HuggingFace Hub.\n"
            "  Check that the repo exists at https://huggingface.co/datasets/anonedgegen/anonedgegen"
        )
    except HfHubHTTPError as e:
        sys.exit(f"Error: Download failed ({e.response.status_code}): {e}")
    except Exception as e:
        sys.exit(f"Error: Download failed: {e}")

    print(f"Done. Files saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
