"""
download_idc.py — 從 NCI Imaging Data Commons（IDC，公開 AWS/GCS bucket，免帳號）下載 CT series

比 TCIA NBIA API 快很多（s5cmd 平行下載），目錄結構可直接給 pylidc 使用：
    <out>/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm

用法：
    python -m scripts.download_idc --collection lidc_idri --out /workspace/data/lidc_dicom
    python -m scripts.download_idc --collection lidc_idri --out /tmp/x --limit 3   # 試跑
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="lidc_idri")
    ap.add_argument("--modality", default="CT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from idc_index import IDCClient
    c = IDCClient()
    df = c.sql_query(f"""
        SELECT SeriesInstanceUID, series_size_MB
        FROM index
        WHERE collection_id = '{args.collection}' AND Modality = '{args.modality}'
        ORDER BY PatientID, SeriesInstanceUID""")
    if args.limit:
        df = df.head(args.limit)
    print(f"IDC {c.get_idc_version()} {args.collection}/{args.modality}: "
          f"{len(df)} series, {df.series_size_MB.sum() / 1024:.1f} GB", flush=True)
    c.download_from_selection(
        seriesInstanceUID=list(df.SeriesInstanceUID),
        downloadDir=args.out,
        dirTemplate="%PatientID/%StudyInstanceUID/%SeriesInstanceUID",
        show_progress_bar=False,
        quiet=False,
    )


if __name__ == "__main__":
    main()
