#!/usr/bin/env python3
"""Pinned, decoder-only weight similarities. No model execution or full shard download."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import struct
import time

from huggingface_hub import hf_hub_download, hf_hub_url
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def decoder_names(weight_map: dict, prefix: str, suffix: str, depth: int) -> dict[int, str]:
    # Anchored namespace: mtp.layers.0 must never masquerade as decoder layer 0.
    pattern = re.compile(r"^" + re.escape(prefix) + r"(\d+)\." + re.escape(suffix) + r"$")
    result = {}
    for name in weight_map:
        match = pattern.fullmatch(name)
        if match:
            layer = int(match.group(1))
            if not 0 <= layer < depth:
                raise ValueError(f"Decoder layer {layer} outside declared depth {depth}")
            result[layer] = name
    if not result:
        raise ValueError(f"No tensors for {prefix}*.{suffix}")
    return dict(sorted(result.items()))


def decode_tensor(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == "BF16":
        values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    elif dtype == "F16":
        values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "F32":
        values = np.frombuffer(raw, dtype="<f4").copy()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if values.size != int(np.prod(shape)) or not np.isfinite(values).all():
        raise ValueError("Wrong tensor size or nonfinite tensor")
    return values


def read_range(url: str, start: int, end: int) -> bytes:
    # Stream and reject a full-file response before reading its body. These shards are GBs.
    for attempt in range(3):
        try:
            with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                              stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                actual = response.headers.get("Content-Range", "")
                if response.status_code != 206 or not actual.startswith(f"bytes {start}-{end}/"):
                    raise ValueError(f"Server did not honor requested byte range: status {response.status_code}")
                chunks = []
                size = 0
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > end - start + 1:
                        raise ValueError("Range response too large")
                    chunks.append(chunk)
                data = b"".join(chunks)
                if len(data) != end - start + 1:
                    raise ValueError("Truncated range response")
                return data
        except (requests.RequestException, ValueError) as exc:
            if attempt == 2:
                # Avoid exposing temporary signed URLs in network exceptions.
                raise RuntimeError(f"Byte-range request failed ({type(exc).__name__})") from None
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def similarities(tensors: dict[int, np.ndarray], depth: int, period: int,
                 region: tuple[int, int], min_pairs: int) -> tuple[list, list, dict]:
    ids = sorted(tensors)
    sizes = {v.size for v in tensors.values()}
    if len(sizes) != 1:
        raise ValueError("Tensor dimensions differ across layers; choose a homogeneous parameter family")
    matrix = np.stack([tensors[i] for i in ids])
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Zero or nonfinite norm")
    matrix /= norms[:, None]
    gram = np.clip(matrix @ matrix.T, -1.0, 1.0)
    rows = []
    for a, left in enumerate(ids):
        for b in range(a + 1, len(ids)):
            right = ids[b]
            inside = region[0] <= left <= region[1] and region[0] <= right <= region[1]
            rows.append({"layer_a": left, "layer_b": right, "offset": right - left,
                         "cosine": float(gram[a, b]), "both_inside_region": inside})
    spectrum = []
    for offset in range(1, depth):
        selected = [r for r in rows if r["offset"] == offset]
        if selected:
            spectrum.append({"offset": offset, "n_pairs": len(selected),
                             "mean_cosine": float(np.mean([r["cosine"] for r in selected])),
                             "rank_eligible": len(selected) >= min_pairs})
    ranked = sorted((r for r in spectrum if r["rank_eligible"]),
                    key=lambda r: (-r["mean_cosine"], r["offset"]))
    by_period = {}
    for offset in (period, 2 * period):
        selected = [r for r in rows if r["offset"] == offset]
        inside = [r["cosine"] for r in selected if r["both_inside_region"]]
        outside = [r["cosine"] for r in selected if not r["both_inside_region"]]
        by_period[str(offset)] = {
            "n_pairs": len(selected),
            "all_valid_mean": float(np.mean([r["cosine"] for r in selected])) if selected else None,
            "n_inside": len(inside), "inside_mean": float(np.mean(inside)) if inside else None,
            "inside_min": min(inside) if inside else None,
            "inside_max": max(inside) if inside else None,
            "n_outside": len(outside), "outside_mean": float(np.mean(outside)) if outside else None}
    return rows, spectrum, {"top_offsets": ranked[:4], "periods": by_period,
                             "global_pair_mean": float(np.mean([r["cosine"] for r in rows])),
                             "tensor_layers": ids}


def scan_model(spec: dict, cache: Path, provenance: Path, workers: int,
               expected_hashes: dict) -> tuple[list, list, list, dict]:
    repo, revision = spec["repo_id"], spec["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full pinned Hub commit SHA")
    metadata = {}
    for filename in ("config.json", "model.safetensors.index.json"):
        path = Path(hf_hub_download(repo, filename, revision=revision, token=False))
        data = path.read_bytes()
        expected = spec["metadata_sha256"][filename]
        if digest(data) != expected:
            raise ValueError(f"Metadata hash mismatch: {repo}/{filename}")
        (provenance / spec["slug"]).mkdir(parents=True, exist_ok=True)
        (provenance / spec["slug"] / filename).write_bytes(data)
        metadata[filename] = json.loads(data)
    config = metadata["config.json"]
    text = config.get("text_config", config)
    depth = int(text["num_hidden_layers"])
    if depth != spec["depth"]:
        raise ValueError("Pinned config depth mismatch")
    weight_map = metadata["model.safetensors.index.json"]["weight_map"]
    groups = {suffix: decoder_names(weight_map, spec["decoder_prefix"], suffix, depth)
              for suffix in spec["tensors"]}
    all_names = sorted({name for names in groups.values() for name in names.values()})
    # Verify selected tensor family matches the configured attention types when available.
    if "layer_types" in text:
        for suffix, names in groups.items():
            wanted = "linear_attention" if suffix.startswith("linear_attn.") else "full_attention"
            expected_ids = {i for i, kind in enumerate(text["layer_types"]) if kind == wanted}
            if set(names) != expected_ids:
                raise ValueError(f"Tensor coverage differs from config.layer_types: {suffix}")
    headers = {}
    for shard in sorted({weight_map[name] for name in all_names}):
        url = hf_hub_url(repo, shard, revision=revision)
        header_len = struct.unpack("<Q", read_range(url, 0, 7))[0]
        if header_len > 64 * 1024 * 1024:
            raise ValueError("Unreasonable safetensors header")
        raw_header = read_range(url, 8, 7 + header_len)
        headers[shard] = (url, json.loads(raw_header), 8 + header_len, digest(raw_header))

    def fetch(name):
        shard = weight_map[name]
        url, header, base, header_hash = headers[shard]
        entry = header[name]
        a, b = entry["data_offsets"]
        local = cache / spec["slug"] / revision / (digest(name.encode()) + ".bin")
        if local.exists():
            raw = local.read_bytes()
            if len(raw) != b - a:
                raise ValueError("Cache size mismatch; move the bad cache file aside and retry")
        else:
            raw = read_range(url, base + a, base + b - 1)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(raw)
        expected_hash = expected_hashes.get((repo, revision, name))
        if expected_hash and digest(raw) != expected_hash:
            raise ValueError(f"Tensor hash differs from the released capture: {name}")
        values = decode_tensor(raw, entry["dtype"], entry["shape"])
        return name, values, {"tensor_name": name, "shard": shard, "dtype": entry["dtype"],
                              "shape": entry["shape"], "n_bytes": len(raw), "sha256": digest(raw),
                              "shard_header_sha256": header_hash,
                              "absolute_byte_start": base + a, "absolute_byte_end": base + b - 1}

    print(f"Reading {repo}@{revision[:12]}: {len(all_names)} decoder tensors", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fetched = list(pool.map(fetch, all_names))
    values = {name: array for name, array, _ in fetched}
    manifests = [entry for _, _, entry in fetched]
    all_rows, all_offsets, summaries = [], [], []
    for suffix, names in groups.items():
        rows, spectrum, summary = similarities(
            {i: values[name] for i, name in names.items()}, depth, spec["period"],
            tuple(spec["region"]), spec["min_pairs"])
        common = {"model": repo, "model_slug": spec["slug"], "revision": revision, "tensor": suffix}
        all_rows.extend(dict(common, **row) for row in rows)
        all_offsets.extend(dict(common, **row) for row in spectrum)
        summaries.append(dict(common, depth=depth, decoder_prefix=spec["decoder_prefix"],
                              period=spec["period"], region=spec["region"],
                              min_rank_pairs=spec["min_pairs"], **summary))
        lead = summary["top_offsets"][0]
        print(f"  {suffix}: {len(names)}/{depth} blocks; peak {lead['offset']} "
              f"cos={lead['mean_cosine']:.6f}, n={lead['n_pairs']}", flush=True)
    return all_rows, all_offsets, summaries, {"repo_id": repo, "revision": revision,
                                              "selected_tensors": manifests}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "models.lock.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--provenance", type=Path, default=ROOT / "provenance" / "metadata")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache" / "tensors")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    lock_bytes = args.lock.read_bytes()
    lock = json.loads(lock_bytes)
    expected_hashes = {}
    released = ROOT / "data" / "provenance.json"
    if released.exists():
        for model in json.loads(released.read_text())["models"]:
            for tensor in model["selected_tensors"]:
                expected_hashes[(model["repo_id"], model["revision"], tensor["tensor_name"])] = tensor["sha256"]
    rows, offsets, summaries, manifests = [], [], [], []
    started = time.monotonic()
    for spec in lock["models"]:
        a, b, c, d = scan_model(spec, args.cache, args.provenance, args.workers, expected_hashes)
        rows.extend(a); offsets.extend(b); summaries.extend(c); manifests.append(d)
    write_csv(args.out / "pairs.csv", rows)
    write_csv(args.out / "offsets.csv", offsets)
    save_json(args.out / "summary.json", {"schema_version": 1, "analyses": summaries})
    save_json(args.out / "provenance.json", {
        "schema_version": 1, "models_lock_sha256": digest(lock_bytes),
        "scanner_sha256": digest(Path(__file__).read_bytes()),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "requests", "huggingface_hub")},
        "elapsed_seconds": round(time.monotonic() - started, 2), "models": manifests})
    print(f"Saved {len(rows)} layer pairs to {args.out}")


if __name__ == "__main__":
    main()
