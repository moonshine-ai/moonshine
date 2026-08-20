"""Build the no-audio UWB-ATCC session-disjoint split CSV and optionally upload it.

    python examples/python/finetune/publish_uwb_splits.py --output /tmp/uwb_atcc_splits.csv
    python examples/python/finetune/publish_uwb_splits.py --upload
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="uwb_atcc_splits.csv")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="push to moonshine-ai/uwb-atcc-session-disjoint-splits",
    )
    args = parser.parse_args(argv)

    from moonshine_voice.lora.data import UWB_SPLITS_REPO, uwb_split_csv_rows

    rows = uwb_split_csv_rows()
    path = Path(args.output)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "session", "scored", "session_disjoint_train"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(UWB_SPLITS_REPO, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo="uwb_atcc_splits.csv",
            repo_id=UWB_SPLITS_REPO,
            repo_type="dataset",
        )
        readme = path.with_name("README.md")
        readme.write_text(
            "---\nlicense: cc-by-nc-sa-4.0\n---\n\n"
            "# UWB-ATCC session-disjoint splits\n\n"
            "IDs only, no audio. Train drops the one session shared with the "
            "published test split (`TWR-34720N`) so an in-domain number is "
            "domain adaptation, not session leakage. Audio stays on "
            "`Jzuluaga/uwb_atcc` (CC BY-NC-SA 4.0).\n"
        )
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=UWB_SPLITS_REPO,
            repo_type="dataset",
        )
        print(f"uploaded {UWB_SPLITS_REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
