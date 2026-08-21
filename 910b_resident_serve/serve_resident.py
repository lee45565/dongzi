#!/usr/bin/env python3
"""910B resident multi-NPU inference server (mp.spawn).

Replaces one-shot:

    python -m torch.distributed.run --nproc_per_node=4 motion_control.py \\
        --input_image ... --driving_video ...

With:

    1. Parent process: HTTP API, does not occupy NPU.
    2. mp.spawn x N workers: each binds one NPU, HCCL init, load SAM3+DiT once.
    3. Subsequent requests only send image/video/prompt; weights stay in HBM.

Copy this directory next to motion_control.py:

    /data02/lyh/ics2v-new/src/30011_motion_control2v/910b/

Then:

    bash run_serve_910b.sh
    python submit_job.py --image /path/a.jpg --video /path/b.mp4 --prompt "A person is dancing"
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Parent process must NOT import torch / torch_npu / motion_control.
# spawn children re-import this module; keep top-level at stdlib only.


DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = "29511"
DEFAULT_HTTP_PORT = 8088
SHUTDOWN_JOB = {"cmd": "shutdown"}


def _worker_env(rank: int, world_size: int, master_addr: str, master_port: str) -> None:
    os.environ["PLATFORM"] = "ascend_npu"
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["ICS2V_RESIDENT"] = "1"
    # Stop motion_control.py from relaunching torchrun inside the worker.
    os.environ["ICS2V_SKIP_RELAUNCH"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("YOLO_OFFLINE", "1")


def _init_npu(rank: int, world_size: int):
    import torch
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    if hasattr(torch, "npu"):
        torch.npu.set_device(rank)
        try:
            # NZ / internal format: large MatMul win on 910B.
            if hasattr(torch.npu, "config"):
                torch.npu.config.allow_internal_format = True
        except Exception:
            pass

    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        )
    return torch, dist


def _maybe_keep_sam3(engine: Any, keep_sam3: bool) -> None:
    """Resident mode: do not destroy SAM3 weights between requests."""
    if not keep_sam3:
        return
    orig = getattr(engine, "_release_sam3_npu", None)
    if orig is None:
        return

    def _release_keep_weights(*_a, **_k):
        try:
            import gc
            import torch

            gc.collect()
            if hasattr(torch, "npu"):
                torch.npu.empty_cache()
        except Exception:
            pass

    engine._release_sam3_npu = _release_keep_weights


def _build_engine(config_yaml: str, extra_args: list[str]) -> Any:
    """Load VideoPoseInference once. Does not run a job."""
    import argparse as _argparse

    # motion_control.py lives in the 910b directory; make sure it is importable.
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    for p in (here, cwd, here.parent):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    import motion_control as mc

    parser = None
    if hasattr(mc, "parse_args"):
        # Some trees expose parse_args(); we still inject yaml/empty inputs.
        pass
    if hasattr(mc, "build_argparser"):
        parser = mc.build_argparser()
    elif hasattr(mc, "get_parser"):
        parser = mc.get_parser()

    ns = None
    if parser is not None:
        ns = parser.parse_args(
            ["--config_yaml", config_yaml, "--input_image", "", "--driving_video", "", *extra_args]
        )
    else:
        ns = _argparse.Namespace(
            config_yaml=config_yaml,
            input_image="",
            driving_video="",
            prompt="",
            resident=True,
        )
        for i, tok in enumerate(extra_args):
            if tok.startswith("--") and i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                setattr(ns, tok.lstrip("-").replace("-", "_"), extra_args[i + 1])

    if hasattr(mc, "VideoPoseInference"):
        cls = mc.VideoPoseInference
        try:
            engine = cls(ns)
        except TypeError:
            engine = cls(config_yaml=config_yaml)
        return engine

    raise RuntimeError(
        "motion_control.VideoPoseInference not found. "
        "Copy serve_resident.py next to motion_control.py, or expose that class."
    )


def _run_job(engine: Any, image: str, video: str, prompt: str) -> dict[str, Any]:
    """Call whatever generate/infer API the current tree has."""
    for name in ("generate", "infer", "run", "process", "__call__"):
        fn = getattr(engine, name, None)
        if not callable(fn) or name == "__call__":
            continue
        try:
            out = fn(image, video, prompt)
            return _normalize_result(out, image=image, video=video)
        except TypeError:
            try:
                out = fn(input_image=image, driving_video=video, prompt=prompt)
                return _normalize_result(out, image=image, video=video)
            except TypeError:
                continue

    # Fallback: set attributes then call a no-arg run.
    for attr, val in (
        ("input_image", image),
        ("driving_video", video),
        ("prompt", prompt),
        ("ref_image", image),
        ("pose_video", video),
    ):
        if hasattr(engine, attr):
            setattr(engine, attr, val)
    for name in ("generate", "infer", "run", "process"):
        fn = getattr(engine, name, None)
        if callable(fn):
            out = fn()
            return _normalize_result(out, image=image, video=video)

    raise RuntimeError(
        "VideoPoseInference has no reusable generate/infer/run method. "
        "Add generate(input_image, driving_video, prompt) and skip torchrun relaunch "
        "when ICS2V_RESIDENT=1."
    )


def _normalize_result(out: Any, image: str, video: str) -> dict[str, Any]:
    if isinstance(out, dict):
        out.setdefault("code", 0)
        out.setdefault("msg", "success")
        return out
    if isinstance(out, str):
        return {"code": 0, "msg": "success", "output": out, "image": image, "video": video}
    if out is None:
        return {"code": 0, "msg": "success", "output": None, "image": image, "video": video}
    return {"code": 0, "msg": "success", "output": str(out)}


def worker_main(
    rank: int,
    world_size: int,
    req_queue,
    res_queue,
    ready_queue,
    worker_args: dict[str, Any],
) -> None:
    t0 = time.time()
    _worker_env(
        rank,
        world_size,
        worker_args["master_addr"],
        worker_args["master_port"],
    )
    try:
        torch, dist = _init_npu(rank, world_size)
        engine = _build_engine(worker_args["config_yaml"], worker_args.get("extra_args") or [])
        _maybe_keep_sam3(engine, keep_sam3=bool(worker_args.get("keep_sam3", True)))
        dist.barrier()
        ready_queue.put(
            {
                "rank": rank,
                "ok": True,
                "load_s": round(time.time() - t0, 3),
                "device": f"npu:{rank}",
            }
        )
        print(
            f"[resident] rank{rank} models loaded in {time.time() - t0:.1f}s, waiting for jobs",
            flush=True,
        )
    except Exception as exc:
        ready_queue.put({"rank": rank, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()
        return

    while True:
        job = None
        if rank == 0:
            job = req_queue.get()
            payload = [job]
        else:
            payload = [None]
        dist.broadcast_object_list(payload, src=0)
        job = payload[0]
        if not job or job.get("cmd") == "shutdown":
            break
        image = job["image"]
        video = job["video"]
        prompt = job.get("prompt") or ""
        t1 = time.time()
        try:
            if rank == 0:
                print(
                    f"[resident] job {job.get('job_id', '-')} image={image} video={video}",
                    flush=True,
                )
            result = _run_job(engine, image, video, prompt)
            result["elapsed_s"] = round(time.time() - t1, 3)
            result["job_id"] = job.get("job_id")
            if rank == 0:
                res_queue.put(result)
        except Exception as exc:
            traceback.print_exc()
            if rank == 0:
                res_queue.put(
                    {
                        "code": 1,
                        "msg": f"{type(exc).__name__}: {exc}",
                        "job_id": job.get("job_id"),
                        "elapsed_s": round(time.time() - t1, 3),
                    }
                )
        dist.barrier()

    try:
        dist.destroy_process_group()
    except Exception:
        pass


class _Handler(BaseHTTPRequestHandler):
    server_version = "ics2v-resident/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[http] " + (fmt % args) + "\n")

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        ctx = self.server.ctx  # type: ignore[attr-defined]
        if path in ("/health", "/"):
            self._json(200, {"status": "ok", "ready": ctx["ready"].is_set(), "world_size": ctx["world_size"]})
            return
        if path == "/ready":
            if ctx["ready"].is_set():
                self._json(200, {"status": "ready", "load": ctx.get("load_info")})
            else:
                self._json(503, {"status": "loading"})
            return
        self._json(404, {"code": 1, "msg": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        ctx = self.server.ctx  # type: ignore[attr-defined]
        if path not in ("/generate", "/v1/generate"):
            self._json(404, {"code": 1, "msg": "not found"})
            return
        if not ctx["ready"].is_set():
            self._json(503, {"code": 1, "msg": "models not ready"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"code": 1, "msg": "body must be JSON"})
            return
        image = data.get("image") or data.get("input_image") or data.get("ref_image")
        video = data.get("video") or data.get("driving_video") or data.get("pose_video")
        prompt = data.get("prompt") or ""
        if not image or not video:
            self._json(400, {"code": 1, "msg": "JSON must include image and video paths"})
            return
        if not os.path.isfile(image):
            self._json(400, {"code": 1, "msg": f"image not found: {image}"})
            return
        if not os.path.isfile(video):
            self._json(400, {"code": 1, "msg": f"video not found: {video}"})
            return

        job = {
            "cmd": "generate",
            "job_id": f"{int(time.time() * 1000)}",
            "image": os.path.abspath(image),
            "video": os.path.abspath(video),
            "prompt": prompt,
        }
        with ctx["lock"]:
            ctx["req_queue"].put(job)
            try:
                result = ctx["res_queue"].get(timeout=ctx["timeout_s"])
            except Exception:
                self._json(504, {"code": 1, "msg": f"timeout after {ctx['timeout_s']}s", "job_id": job["job_id"]})
                return
        http_code = 200 if int(result.get("code", 1)) == 0 else 500
        self._json(http_code, result)


def _wait_ready(ready_queue, world_size: int, timeout_s: float) -> list[dict[str, Any]]:
    infos = []
    deadline = time.time() + timeout_s
    while len(infos) < world_size:
        remain = deadline - time.time()
        if remain <= 0:
            raise TimeoutError(
                f"only {len(infos)}/{world_size} workers became ready within {timeout_s}s: {infos}"
            )
        infos.append(ready_queue.get(timeout=remain))
    failed = [x for x in infos if not x.get("ok")]
    if failed:
        raise RuntimeError(f"worker load failed: {failed}")
    return infos


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="910B mp.spawn resident inference server")
    p.add_argument("--config_yaml", default="motion_control.yaml")
    p.add_argument("--nproc", type=int, default=int(os.environ.get("ICS2V_NPROC", "4")))
    p.add_argument("--host", default=os.environ.get("ICS2V_SERVE_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("ICS2V_SERVE_PORT", str(DEFAULT_HTTP_PORT))))
    p.add_argument("--master_addr", default=os.environ.get("MASTER_ADDR", DEFAULT_MASTER_ADDR))
    p.add_argument("--master_port", default=os.environ.get("MASTER_PORT", DEFAULT_MASTER_PORT))
    p.add_argument("--timeout_s", type=float, default=float(os.environ.get("ICS2V_JOB_TIMEOUT", "3600")))
    p.add_argument("--load_timeout_s", type=float, default=float(os.environ.get("ICS2V_LOAD_TIMEOUT", "900")))
    p.add_argument(
        "--keep_sam3",
        action="store_true",
        default=True,
        help="Keep SAM3 weights resident (default). Disable if DiT OOMs.",
    )
    p.add_argument("--no_keep_sam3", action="store_false", dest="keep_sam3")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded toward motion_control")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extra = list(args.extra or [])
    if extra and extra[0] == "--":
        extra = extra[1:]

    # Parent stays torch-free (no NPU bind). Workers are spawn processes, same
    # contract as torch.multiprocessing.spawn(nprocs=N, join=False).
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    ctx_mp = mp.get_context("spawn")

    world_size = int(args.nproc)
    req_queue = ctx_mp.Queue()
    res_queue = ctx_mp.Queue()
    ready_queue = ctx_mp.Queue()
    worker_args = {
        "config_yaml": args.config_yaml,
        "master_addr": args.master_addr,
        "master_port": str(args.master_port),
        "keep_sam3": bool(args.keep_sam3),
        "extra_args": extra,
    }

    print(
        f"[resident] spawning {world_size} workers, HTTP {args.host}:{args.port}, "
        f"HCCL {args.master_addr}:{args.master_port}",
        flush=True,
    )
    workers: list[mp.Process] = []
    for rank in range(world_size):
        proc = ctx_mp.Process(
            target=worker_main,
            name=f"ics2v-rank{rank}",
            args=(rank, world_size, req_queue, res_queue, ready_queue, worker_args),
            daemon=False,
        )
        proc.start()
        workers.append(proc)

    try:
        load_info = _wait_ready(ready_queue, world_size, args.load_timeout_s)
        print(f"[resident] all workers ready: {load_info}", flush=True)
    except Exception:
        traceback.print_exc()
        _shutdown_workers(req_queue, workers)
        return 1

    http_ctx = {
        "req_queue": req_queue,
        "res_queue": res_queue,
        "lock": threading.Lock(),
        "ready": threading.Event(),
        "world_size": world_size,
        "timeout_s": args.timeout_s,
        "load_info": load_info,
    }
    http_ctx["ready"].set()

    httpd = ThreadingHTTPServer((args.host, int(args.port)), _Handler)
    httpd.ctx = http_ctx  # type: ignore[attr-defined]
    print(
        f"[resident] listening on http://{args.host}:{args.port}  "
        f"POST /generate  JSON {{\"image\":\"...\",\"video\":\"...\",\"prompt\":\"...\"}}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[resident] shutting down", flush=True)
    finally:
        httpd.server_close()
        _shutdown_workers(req_queue, workers)
    return 0


def _shutdown_workers(req_queue, workers: list[mp.Process]) -> None:
    try:
        req_queue.put(SHUTDOWN_JOB)
    except Exception:
        pass
    for proc in workers:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
