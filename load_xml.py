#!/usr/bin/env python3
"""
Concatenate MJCF (XML) humanoid templates into two files:
- first 52 (usable) templates -> out_good
- remaining 12 (faulty; lines prefixed with "#####") -> out_faulty

Templates are separated by: 3 empty lines, six hyphens, 3 empty lines
i.e., "\n\n\n------\n\n\n"
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

TEMPLATE_LIST = r"""
- mjcf/smpl/a0f02530_smpl.xml
- mjcf/smpl/b156eebd_smpl.xml
- mjcf/smpl/584b793a_smpl.xml
- mjcf/smpl/aef91182_smpl.xml
- mjcf/smpl/0c3f729e_smpl.xml
- mjcf/smpl/1f0234a6_smpl.xml
- mjcf/smpl/aaab922b_smpl.xml
- mjcf/smpl/97d89d08_smpl.xml

- mjcf/smpl/f16065b0_smpl.xml
- mjcf/smpl/5dcdf59a_smpl.xml
- mjcf/smpl/f00a7a9f_smpl.xml
- mjcf/smpl/25bef108_smpl.xml
- mjcf/smpl/85c00aec_smpl.xml
- mjcf/smpl/5636a12a_smpl.xml
- mjcf/smpl/33a17abb_smpl.xml
- mjcf/smpl/124f1e57_smpl.xml

- mjcf/smpl/e585e9fd_smpl.xml
- mjcf/smpl/520ed34f_smpl.xml
- mjcf/smpl/5dbdeb54_smpl.xml
- mjcf/smpl/65f23504_smpl.xml
- mjcf/smpl/948528ba_smpl.xml
- mjcf/smpl/f0f7976f_smpl.xml
- mjcf/smpl/20cf78b3_smpl.xml
- mjcf/smpl/f56ef3f7_smpl.xml

- mjcf/smpl/fba2c39a_smpl.xml
- mjcf/smpl/b80aed9f_smpl.xml
- mjcf/smpl/706ed6c9_smpl.xml
- mjcf/smpl/39f19cab_smpl.xml
- mjcf/smpl/fcc491cd_smpl.xml
- mjcf/smpl/bc793600_smpl.xml
- mjcf/smpl/74fc526e_smpl.xml
- mjcf/smpl/09016021_smpl.xml

- mjcf/smpl/6895f004_smpl.xml
- mjcf/smpl/602dbc36_smpl.xml
- mjcf/smpl/22e9a6f8_smpl.xml
- mjcf/smpl/85011266_smpl.xml
- mjcf/smpl/9f630f94_smpl.xml
- mjcf/smpl/dc46f761_smpl.xml
- mjcf/smpl/c51c47d2_smpl.xml
- mjcf/smpl/3577c351_smpl.xml

- mjcf/smpl/18ce6b2c_smpl.xml
- mjcf/smpl/ef892c76_smpl.xml
- mjcf/smpl/349bdc0e_smpl.xml
- mjcf/smpl/6803e1fa_smpl.xml
- mjcf/smpl/e9f8d7a4_smpl.xml
- mjcf/smpl/8a24b3b7_smpl.xml
- mjcf/smpl/00c972db_smpl.xml
- mjcf/smpl/6046abb1_smpl.xml

- mjcf/smpl/76144ae7_smpl.xml
- mjcf/smpl/e698f1e8_smpl.xml
- mjcf/smpl/b944e212_smpl.xml
- mjcf/smpl/2a31c8ac_smpl.xml

##### - mjcf/smpl/7e43c211_smpl.xml
##### - mjcf/smpl/b3428686_smpl.xml
##### - mjcf/smpl/0e091a72_smpl.xml
##### - mjcf/smpl/aca66100_smpl.xml
##### - mjcf/smpl/af4dbe08_smpl.xml
##### - mjcf/smpl/75e01b05_smpl.xml
##### - mjcf/smpl/31f56211_smpl.xml
##### - mjcf/smpl/0f637664_smpl.xml
##### - mjcf/smpl/ef483c8a_smpl.xml
##### - mjcf/smpl/15c4d2e9_smpl.xml
##### - mjcf/smpl/0a1ece18_smpl.xml
##### - mjcf/smpl/638a4fb7_smpl.xml
""".strip("\n")


SEP = "\n\n\n------\n\n\n"


def parse_templates(text: str) -> tuple[list[str], list[str]]:
    good, faulty = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#####"):
            # e.g. "##### - mjcf/smpl/xxx.xml"
            rest = line.lstrip("#").strip()
            if rest.startswith("-"):
                faulty.append(rest[1:].strip())
            continue
        if line.startswith("-"):
            good.append(line[1:].strip())
    return good, faulty


def read_files(asset_root: Path, rel_paths: list[str]) -> list[str]:
    contents: list[str] = []
    for rel in rel_paths:
        p = asset_root / rel
        try:
            contents.append(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"[WARN] Missing file: {p}", file=sys.stderr)
        except UnicodeDecodeError:
            # MJCF is usually UTF-8; fall back if needed.
            contents.append(p.read_text(encoding="utf-8", errors="replace"))
    return contents


def write_concat(out_path: Path, chunks: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(SEP.join(chunks), encoding="utf-8")



def xml_to_betas_path(xml_rel: str) -> str:
    # "mjcf/smpl/xxxx_smpl.xml" -> "mjcf/smpl/xxxx_betas.pt"
    if not xml_rel.endswith("_smpl.xml"):
        raise ValueError(f"Unexpected template name (expected *_smpl.xml): {xml_rel}")
    return xml_rel[:-len("_smpl.xml")] + "_betas.pt"


def read_betas(asset_root: Path, xml_rel_paths: list[str]):
    """
    Load betas tensors corresponding to the given XML templates.
    Returns a list; missing files are skipped with a warning (same behavior as read_files()).
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("read_betas requires PyTorch (import torch failed).") from e

    betas = []
    for xml_rel in xml_rel_paths:
        betas_rel = xml_to_betas_path(xml_rel)
        p = asset_root / betas_rel
        try:
            obj = torch.load(p, map_location="cpu", weights_only=True)
            # accept tensor or dict-like containers (common in checkpoints)
            if isinstance(obj, dict):
                # try a few conventional keys; fall back to first tensor value
                for k in ("betas", "beta", "shape", "shapes"):
                    if k in obj:
                        obj = obj[k]
                        break
                else:
                    for v in obj.values():
                        if torch.is_tensor(v):
                            obj = v
                            break
            if not torch.is_tensor(obj):
                raise TypeError(f"Loaded object is not a tensor (or tensor-containing dict): {type(obj)}")
            betas.append(obj)
        except FileNotFoundError:
            print(f"[WARN] Missing betas file: {p}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Failed to load betas from {p}: {e}", file=sys.stderr)
    return betas


def load_betas_split(asset_root: Path, template_list_text: str):
    """
    Uses your parse_templates() to split into first 52 good and 12 faulty,
    then loads betas accordingly. Returns (good_betas, faulty_betas).
    """
    good, faulty = parse_templates(template_list_text)
    good_52 = good[:52]
    faulty_12 = faulty[:12]
    return read_betas(asset_root, good_52), read_betas(asset_root, faulty_12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-root", type=Path, default=Path("ase/data/assets"),
                    help="Root directory that contains mjcf/...")
    ap.add_argument("--out-good", type=Path, default=Path("smpl_templates_good_52.xml"),
                    help="Output file for the first 52 usable templates")
    ap.add_argument("--out-faulty", type=Path, default=Path("smpl_templates_faulty_12.xml"),
                    help="Output file for the 12 faulty templates")
    args = ap.parse_args()

    good, faulty = parse_templates(TEMPLATE_LIST)

    # Enforce the requested split: first 52 vs remaining 12
    good_52 = good[:52]
    faulty_12 = faulty[:12]

    good_contents = read_files(args.asset_root, good_52)
    faulty_contents = read_files(args.asset_root, faulty_12)

    write_concat(args.out_good, good_contents)
    write_concat(args.out_faulty, faulty_contents)

    print(f"[OK] Wrote {len(good_contents)} templates -> {args.out_good}")
    print(f"[OK] Wrote {len(faulty_contents)} templates -> {args.out_faulty}")

    good_betas = read_betas(args.asset_root, good_52)
    faulty_betas = read_betas(args.asset_root, faulty_12)
    # print(f"[OK] Loaded {len(good_betas)} betas (good)")
    # print(f"[OK] Loaded {len(faulty_betas)} betas (faulty)")

    print(good_betas)
    print(faulty_betas)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


