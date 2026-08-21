"""Required hooks inside motion_control.py for resident mp.spawn.

This file is documentation-as-code. Paste the snippets into motion_control.py
on the 910B tree; do not import this module at runtime.

1) Skip torchrun / distributed.run relaunch when already spawned
----------------------------------------------------------------
Near the existing multi-card relaunch (the block that execs torch.distributed.run
or sets RANK):

    import os
    if os.environ.get("ICS2V_RESIDENT") == "1" or os.environ.get("ICS2V_SKIP_RELAUNCH") == "1":
        # Parent already did mp.spawn + HCCL init. Do not launch again.
        pass
    elif not torch.distributed.is_initialized() and gpu_count > 1:
        ... original relaunch ...

2) Expose a reusable generate() that does NOT load models again
---------------------------------------------------------------
VideoPoseInference.__init__ should only load SAM3 + DiT.
Move the current one-shot path (Step1 mask + Step2 DiT) into:

    def generate(self, input_image: str, driving_video: str, prompt: str = "") -> dict:
        # existing Step 0 / 1 / 2 using the given paths
        return {"code": 0, "msg": "success", "output": output_mp4}

main() for one-shot CLI can stay:

    engine = VideoPoseInference(args)
    if os.environ.get("ICS2V_RESIDENT") == "1":
        return  # server worker keeps engine
    result = engine.generate(args.input_image, args.driving_video, args.prompt)

3) SAM3 between requests
------------------------
_release_sam3_npu() currently unloads SAM3 to free HBM for DiT.
serve_resident.py monkeypatches it to empty_cache only (keep weights).
If 720p DiT OOMs, start the server with --no_keep_sam3 so SAM3 is freed
after each mask; next request will need SAM3 reloaded (seconds, not minutes).

4) Do not combine launchers
---------------------------
Resident:  bash run_serve_910b.sh
One-shot:  bash run_910b.sh image video prompt
Never torch.distributed.run AND mp.spawn at the same time.
"""
