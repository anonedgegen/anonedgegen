import argparse
import os
import pathlib
import sys

from huggingface_hub import HfApi, upload_folder
from huggingface_hub.errors import HfHubHTTPError

TARGET_REPO = "anonedgegen/anonedgegen"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a local folder to the anonedgegen HuggingFace dataset (requires write token)."
    )
    parser.add_argument(
        "folder_path",
        nargs="?",
        default=".",
        help="Local folder to upload (default: current directory)",
    )
    parser.add_argument(
        "--token",
        help="HuggingFace write token. Falls back to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--folder",
        default=None,
        help="Destination folder inside the repo, e.g. --folder my_folder",
    )
    parser.add_argument(
        "--message",
        default="Upload dataset",
        help="Commit message (default: 'Upload dataset')",
    )
    return parser.parse_args()


def resolve_token(args):
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(
            "Error: HuggingFace write token required.\n"
            "  Set the HF_TOKEN environment variable or pass --token <your_token>.\n"
            "  Get a token at https://huggingface.co/settings/tokens"
        )
    return token


def validate_token(token):
    api = HfApi(token=token)
    try:
        info = api.whoami()
    except HfHubHTTPError as e:
        if e.response.status_code in (401, 403):
            sys.exit("Error: Token is invalid or expired. Check your HuggingFace token.")
        sys.exit(f"Error: Token validation failed: {e}")
    except Exception as e:
        sys.exit(f"Error: Could not validate token (network issue?): {e}")
    return info["name"]


def main():
    args = parse_args()

    token = resolve_token(args)
    username = validate_token(token)
    print(f"Authenticated as: {username}")

    folder = pathlib.Path(args.folder_path)
    if not folder.exists():
        sys.exit(f"Error: Folder not found: {args.folder_path}")
    if not folder.is_dir():
        sys.exit(f"Error: Path is not a directory: {args.folder_path}")

    dest = f"uploading to {TARGET_REPO}/{args.folder}" if args.folder else f"uploading to {TARGET_REPO}"
    print(f"Uploading '{folder.resolve()}' — {dest} ...")

    try:
        upload_folder(
            folder_path=str(folder),
            repo_id=TARGET_REPO,
            repo_type="dataset",
            token=token,
            path_in_repo=args.folder,
            commit_message=args.message,
        )
    except HfHubHTTPError as e:
        if e.response.status_code == 403:
            sys.exit(
                "Error: Your token does not have write permission for this repository.\n"
                "  Make sure your token has 'write' access at https://huggingface.co/settings/tokens"
            )
        if e.response.status_code == 401:
            sys.exit("Error: Unauthorized. Your token may have been revoked.")
        sys.exit(f"Error: Upload failed ({e.response.status_code}): {e}")
    except Exception as e:
        sys.exit(f"Error: Upload failed: {e}")

    print(f"Done. Dataset available at: https://huggingface.co/datasets/{TARGET_REPO}")


if __name__ == "__main__":
    main()
