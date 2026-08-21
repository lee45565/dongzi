"""Runtime patches so VideoPoseInference can serve many jobs without dying.

The original class already has do_inference() and loads models in __init__.
These three behaviors break a resident server:

1. _release_sam3_npu() sets predictors to None → second job crashes.
2. Rank-0 early return / sys.exit(1) on bad input → other ranks hang on barrier.
3. Fixed /tmp/_videopose_paths.json is fine for serial jobs; keep it.

apply_resident_compat(engine) is called once after VideoPoseInference().
"""

from __future__ import annotations

import json
import os
import shutil
import traceback
from datetime import datetime
from time import time
from types import MethodType
from typing import Any


def apply_resident_compat(engine: Any, keep_sam3: bool = True) -> Any:
    engine._resident_keep_sam3 = bool(keep_sam3)
    engine._release_sam3_npu = MethodType(_release_sam3_npu_keep, engine)
    engine._restore_sam3_npu = MethodType(_restore_sam3_npu, engine)
    engine._bcast_obj = MethodType(_bcast_obj, engine)
    engine.do_inference = MethodType(_do_inference_resident, engine)
    return engine


def _bcast_obj(self, obj: Any, src: int = 0) -> Any:
    import torch.distributed as dist

    payload = [obj]
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=src)
        return payload[0]
    return obj


def _npu_index() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def _release_sam3_npu_keep(self) -> None:
    """Move SAM3 to CPU to free HBM for DiT; keep Python refs for the next job."""
    import torch

    if not getattr(self, "_resident_keep_sam3", True):
        for attr in ("sam3_video_predictor", "sam3_image_predictor"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            for candidate in (obj, getattr(obj, "model", None)):
                if candidate is None:
                    continue
                try:
                    if hasattr(candidate, "cpu"):
                        candidate.cpu()
                    elif hasattr(candidate, "to"):
                        candidate.to("cpu")
                except Exception:
                    pass
            setattr(self, attr, None)
        self._sam3_available = False
        try:
            torch.npu.empty_cache()
        except Exception:
            pass
        return

    for attr in ("sam3_video_predictor", "sam3_image_predictor"):
        obj = getattr(self, attr, None)
        if obj is None:
            continue
        for candidate in (obj, getattr(obj, "model", None)):
            if candidate is None:
                continue
            try:
                if hasattr(candidate, "cpu"):
                    candidate.cpu()
                elif hasattr(candidate, "to"):
                    candidate.to("cpu")
            except Exception:
                pass
    try:
        torch.npu.empty_cache()
    except Exception:
        pass


def _restore_sam3_npu(self) -> None:
    """Put SAM3 back on this rank's NPU before the next mask pass."""
    if not getattr(self, "_sam3_available", False):
        return
    device = f"npu:{_npu_index()}"
    for attr in ("sam3_video_predictor", "sam3_image_predictor"):
        obj = getattr(self, attr, None)
        if obj is None:
            continue
        for candidate in (obj, getattr(obj, "model", None)):
            if candidate is None:
                continue
            try:
                if hasattr(candidate, "npu"):
                    candidate.npu(_npu_index())
                elif hasattr(candidate, "to"):
                    candidate.to(device)
            except Exception:
                pass


def _do_inference_resident(
    self,
    input_image_path: str,
    driving_video_path: str,
    prompt: str = "",
    negative_prompt: str = None,
    seed: int = None,
    output_video_path: str = "",
    keep_tmp_res: bool = None,
    name: str = None,
    skip_mask_preprocessing: bool = None,
    rendered_video_path: str = "",
    rendered_mask_path: str = "",
    ref_mask_path: str = "",
):
    """Same contract as VideoPoseInference.do_inference, safe for many jobs."""
    import torch.distributed as dist
    from PIL import Image

    from lightx2v.utils.utils import is_main_process
    from motion_control import _NoSubjectError

    if seed is None:
        seed = self.seed
    if negative_prompt is None:
        negative_prompt = self.negative_prompt or self._negative_prompt
    if keep_tmp_res is None:
        keep_tmp_res = self.keep_tmp
    if skip_mask_preprocessing is None:
        skip_mask_preprocessing = self.skip_mask
    output_video_path = output_video_path or self.output

    _verbose = not dist.is_initialized() or is_main_process()
    _is_dist = dist.is_initialized()

    if _is_dist:
        dist.barrier()

    _paths_file = os.path.join(self.temp_root_dir, "_videopose_paths.json")
    os.makedirs(self.temp_root_dir, exist_ok=True)
    if _verbose:
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        if not output_video_path:
            output_video_path = os.path.join(
                self.temp_root_dir, "output_video", f"{name}_videopose_output.mp4"
            )
        os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
        temp_dir = os.path.join(self.temp_root_dir, name)
        counter = 0
        while os.path.isdir(temp_dir):
            counter += 1
            temp_dir = os.path.join(self.temp_root_dir, f"{name}_{counter}")
        os.makedirs(temp_dir, exist_ok=False)
        with open(_paths_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "temp_dir": temp_dir,
                    "output_video_path": output_video_path,
                    "name": name,
                },
                f,
            )

    if _is_dist:
        dist.barrier()
        if not is_main_process():
            with open(_paths_file, "r", encoding="utf-8") as f:
                p = json.load(f)
            temp_dir = p["temp_dir"]
            output_video_path = p["output_video_path"]
            name = p["name"]

    _mask_rendered = rendered_video_path or os.path.join(temp_dir, "rendered_v2.mp4")
    _mask_mask = rendered_mask_path or os.path.join(temp_dir, "rendered_mask_v2.mp4")
    _mask_ref = ref_mask_path or os.path.join(temp_dir, "ref_mask.jpg")

    t_start = time()
    step = {
        "code": 0,
        "msg": "success",
        "driving": driving_video_path,
        "prompt": prompt,
    }

    try:
        if not skip_mask_preprocessing:
            if _verbose:
                try:
                    self._restore_sam3_npu()
                    if not self._sam3_available or getattr(self, "sam3_video_predictor", None) is None:
                        step = {
                            "code": -11,
                            "msg": "SAM3 unavailable.",
                            "driving": driving_video_path,
                            "prompt": prompt,
                        }
                    else:
                        res, msg = self._check_inputs(input_image_path, driving_video_path)
                        if res != 0:
                            step = {
                                "code": res,
                                "msg": msg,
                                "driving": driving_video_path,
                                "prompt": prompt,
                            }
                        else:
                            _npath = os.path.join(temp_dir, "_driving_normalized.mp4")
                            try:
                                driving_video_path, _, nmsg = self._normalize_video(
                                    driving_video_path, _npath
                                )
                            except RuntimeError as e:
                                step = {
                                    "code": -2,
                                    "msg": f"Normalize failed: {e}",
                                    "driving": driving_video_path,
                                    "prompt": prompt,
                                }
                            else:
                                print(f"[Step 0] Video: {nmsg}")
                                t1 = time()
                                print(f"\n{'=' * 60}")
                                print("[Step 1/2] Mask preprocessing (SAM3 + e2e)")
                                print(f"{'=' * 60}")
                                self._preprocess_masks(
                                    ref_image_path=input_image_path,
                                    driving_video_path=driving_video_path,
                                    temp_dir=temp_dir,
                                )
                                print(f"[Step 1/2] Done ({time() - t1:.1f}s)")
                                step = {
                                    "code": 0,
                                    "msg": "success",
                                    "driving": driving_video_path,
                                    "prompt": prompt,
                                }
                except _NoSubjectError as e:
                    print(f"\n[ABORT] {e}")
                    step = {
                        "code": int(e.error_code),
                        "msg": str(e),
                        "driving": driving_video_path,
                        "prompt": prompt,
                    }
                except Exception as e:
                    traceback.print_exc()
                    step = {
                        "code": -10,
                        "msg": str(e),
                        "driving": driving_video_path,
                        "prompt": prompt,
                    }
            step = self._bcast_obj(step)
            if int(step["code"]) != 0:
                return step["code"], step["msg"], output_video_path
            driving_video_path = step.get("driving") or driving_video_path
            prompt = step.get("prompt") or prompt
        else:
            if _verbose:
                print("[Step 1/2] Using pre-computed masks")
                for pth in (_mask_rendered, _mask_mask, _mask_ref):
                    if not os.path.isfile(pth):
                        step = {"code": -1, "msg": f"Missing: {pth}", "driving": driving_video_path, "prompt": prompt}
                        break
            step = self._bcast_obj(step)
            if int(step["code"]) != 0:
                return step["code"], step["msg"], output_video_path

        self._release_sam3_npu()
        if _is_dist:
            dist.barrier()

        if _verbose and self.prompt_enhance and self.prompt_expander.is_available:
            try:
                ref_img = Image.open(input_image_path).convert("RGB")
                expanded = self.prompt_expander(
                    prompt,
                    image=ref_img,
                    seed=seed,
                    tar_lang=self.prompt_expander.tar_lang,
                )
                if expanded.prompt != prompt:
                    print(
                        f"[Prompt] Enhanced: \"{prompt[:60]}...\" "
                        f"→ \"{expanded.prompt[:80]}...\""
                    )
                    prompt = expanded.prompt
            except Exception as e:
                print(f"[Prompt] Enhancement failed: {e}")
        prompt = self._bcast_obj(prompt)

        t1 = time()
        if _verbose:
            print(f"\n{'=' * 60}")
            print("[Step 2/2] Video generation")
            print(f"{'=' * 60}")
            print(f"  prompt: {prompt[:100]}...")

        self.pipeline.pose_path = _mask_rendered
        self.pipeline.image_mask_path = _mask_ref
        self.pipeline.driving_mask_path = _mask_mask

        self.pipeline.generate(
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            save_result_path=output_video_path,
            task=self.task,
            image_path=input_image_path,
        )

        t2 = time()
        if _verbose:
            print(f"[Step 2/2] Video generation done ({t2 - t1:.1f}s)")
            print(f"\n{'=' * 60}")
            print(f"Total inference time: {time() - t_start:.1f}s")
            print(f"Output: {output_video_path}")
            print(f"{'=' * 60}")

    except Exception as e:
        traceback.print_exc()
        if _verbose and not keep_tmp_res and temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
        fail = self._bcast_obj({"code": -10, "msg": str(e)})
        return fail["code"], fail["msg"], output_video_path

    if _verbose and not keep_tmp_res and temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)

    return 0, "success", output_video_path
