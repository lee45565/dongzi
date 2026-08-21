#!/usr/bin/env python3
"""Submit one generate job to the resident server."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("ICS2V_SERVE_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("ICS2V_SERVE_PORT", "8088")))
    p.add_argument("--image", "--input_image", dest="image", required=True)
    p.add_argument("--video", "--driving_video", dest="video", required=True)
    p.add_argument("--prompt", default="A person is dancing")
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()

    url = f"http://{args.host}:{args.port}/generate"
    payload = {
        "image": os.path.abspath(args.image),
        "video": os.path.abspath(args.video),
        "prompt": args.prompt,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body = resp.read().decode("utf-8")
            print(body)
            data = json.loads(body)
            return 0 if int(data.get("code", 1)) == 0 else 1
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
