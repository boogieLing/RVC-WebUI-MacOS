#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from urllib import error as urlerror
from urllib import request as urlrequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge helper for the local Phase 1 API.")
    parser.add_argument("--base-url", required=True)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("refresh")

    select_parser = subparsers.add_parser("select-model")
    select_parser.add_argument("--name", required=True)

    subparsers.add_parser("unload-model")

    subparsers.add_parser("uvr-models")

    uvr_convert_parser = subparsers.add_parser("uvr-convert")
    uvr_convert_parser.add_argument("--request-json", required=True)

    subparsers.add_parser("asset-check")
    subparsers.add_parser("asset-download")

    export_onnx_parser = subparsers.add_parser("export-onnx")
    export_onnx_parser.add_argument("--request-json", required=True)

    ckpt_compare_parser = subparsers.add_parser("ckpt-compare")
    ckpt_compare_parser.add_argument("--request-json", required=True)

    ckpt_show_parser = subparsers.add_parser("ckpt-show")
    ckpt_show_parser.add_argument("--request-json", required=True)

    ckpt_modify_parser = subparsers.add_parser("ckpt-modify")
    ckpt_modify_parser.add_argument("--request-json", required=True)

    ckpt_merge_parser = subparsers.add_parser("ckpt-merge")
    ckpt_merge_parser.add_argument("--request-json", required=True)

    ckpt_extract_parser = subparsers.add_parser("ckpt-extract")
    ckpt_extract_parser.add_argument("--request-json", required=True)

    single_parser = subparsers.add_parser("convert-single")
    single_parser.add_argument("--request-json", required=True)

    batch_parser = subparsers.add_parser("convert-batch")
    batch_parser.add_argument("--request-json", required=True)

    subparsers.add_parser("realtime-devices")
    subparsers.add_parser("realtime-status")

    realtime_config_parser = subparsers.add_parser("realtime-configure")
    realtime_config_parser.add_argument("--request-json", required=True)

    realtime_start_parser = subparsers.add_parser("realtime-start")
    realtime_start_parser.add_argument("--request-json", required=True)

    subparsers.add_parser("realtime-stop")
    return parser.parse_args()


def http_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urlrequest.Request(url, data=data, headers=headers, method=method)
    with urlrequest.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    request_payload = json.loads(args.request_json) if hasattr(args, "request_json") else None

    try:
        if args.command == "refresh":
            payload = http_json("GET", f"{base_url}/phase1/catalog")
        elif args.command == "select-model":
            payload = http_json("POST", f"{base_url}/phase1/select-model", {"name": args.name})
        elif args.command == "unload-model":
            payload = http_json("POST", f"{base_url}/phase1/unload-model")
        elif args.command == "uvr-models":
            payload = http_json("GET", f"{base_url}/phase1/uvr-models")
        elif args.command == "uvr-convert":
            payload = http_json("POST", f"{base_url}/phase1/uvr-convert", json.loads(args.request_json))
        elif args.command == "asset-check":
            payload = http_json("GET", f"{base_url}/phase1/assets-integrity")
        elif args.command == "asset-download":
            payload = http_json("POST", f"{base_url}/phase1/assets-download")
        elif args.command == "export-onnx":
            payload = http_json("POST", f"{base_url}/phase1/export-onnx", json.loads(args.request_json))
        elif args.command == "ckpt-compare":
            payload = http_json("POST", f"{base_url}/phase1/ckpt-compare", json.loads(args.request_json))
        elif args.command == "ckpt-show":
            payload = http_json("POST", f"{base_url}/phase1/ckpt-show", json.loads(args.request_json))
        elif args.command == "ckpt-modify":
            payload = http_json("POST", f"{base_url}/phase1/ckpt-modify", json.loads(args.request_json))
        elif args.command == "ckpt-merge":
            payload = http_json("POST", f"{base_url}/phase1/ckpt-merge", json.loads(args.request_json))
        elif args.command == "ckpt-extract":
            payload = http_json("POST", f"{base_url}/phase1/ckpt-extract", json.loads(args.request_json))
        elif args.command == "convert-single":
            payload = http_json("POST", f"{base_url}/phase1/convert-single", request_payload)
        elif args.command == "convert-batch":
            payload = http_json("POST", f"{base_url}/phase1/convert-batch", request_payload)
        elif args.command == "realtime-devices":
            payload = http_json("GET", f"{base_url}/phase1/realtime/devices")
        elif args.command == "realtime-status":
            payload = http_json("GET", f"{base_url}/phase1/realtime/status")
        elif args.command == "realtime-configure":
            payload = http_json("POST", f"{base_url}/phase1/realtime/configure", json.loads(args.request_json))
        elif args.command == "realtime-start":
            payload = http_json("POST", f"{base_url}/phase1/realtime/start", json.loads(args.request_json))
        elif args.command == "realtime-stop":
            payload = http_json("POST", f"{base_url}/phase1/realtime/stop")
        else:  # pragma: no cover
            raise RuntimeError(f"Unsupported command: {args.command}")
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore").strip()
        if detail:
            try:
                payload = json.loads(detail)
                if isinstance(payload, dict) and payload.get("detail"):
                    print(json.dumps({"error": payload["detail"]}, ensure_ascii=False), file=sys.stderr)
                    return 1
            except json.JSONDecodeError:
                pass
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
