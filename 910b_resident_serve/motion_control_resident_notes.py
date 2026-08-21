"""What to change in motion_control.py — now that we have the real file.

Short answer: this IS the right file, and most of the work is already done.

Already OK
----------
- Models load once in VideoPoseInference.__init__
- do_inference(image, video, prompt) is reusable (do_batch_inference already loops it)
- gpu_count 1/2/4 + WORLD_SIZE check
- __main__ relaunch is skipped when workers already have WORLD_SIZE (serve_resident sets it)

Must fix for a long-lived server (handled at runtime by resident_compat.py)
--------------------------------------------------------------------------
1) _release_sam3_npu() does setattr(..., None). Second do_inference crashes.
   Fix: move SAM3 to CPU, keep the Python objects, restore to NPU next job.

2) On _NoSubjectError, rank 0 does sys.exit(1). That kills the worker.
   Fix: return error_code like the non-dist path.

3) Rank 0 `return res, msg` on bad input happens BEFORE the mask barrier.
   Ranks 1-3 wait forever. Fix: broadcast a status object, every rank returns.

Do NOT edit LightX2V / attn / yaml for this feature.

How to run
----------
Copy 910b_resident_serve/ next to motion_control.py (or copy the py files into 910b/).
  bash run_serve_910b.sh
  python submit_job.py --image a.jpg --video b.mp4 --prompt "A person is dancing"

One-shot CLI is unchanged:
  bash run_910b.sh image video prompt
"""
