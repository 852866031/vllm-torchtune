"""
Run torchtune 0.6.1's LoRA single-device recipe for ~15 wall-clock seconds,
treat the first 5 s as warmup, and print tokens/second over the remaining
window.

The inner loop mirrors recipes/lora_finetune_single_device.py::train in 0.6.1
but adds (a) a wall-clock cap and (b) per-step (elapsed, num_tokens) samples
so we can compute warmup-excluded throughput.
"""
import importlib.util
import os
import sys
import time
import types

import torch
import torchtune
from omegaconf import OmegaConf

from torchtune import config, training, utils


def _load_recipe_class():
    """torchtune 0.4.0 ships recipes/ alongside the package but blocks
    `import recipes.*`. Load the module directly from its file path."""
    recipe_path = os.path.join(
        os.path.dirname(torchtune.__file__),
        "..",
        "recipes",
        "lora_finetune_single_device.py",
    )
    spec = importlib.util.spec_from_file_location(
        "lora_finetune_single_device_local", recipe_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LoRAFinetuneRecipeSingleDevice


LoRAFinetuneRecipeSingleDevice = _load_recipe_class()

WINDOW_SEC = 60.0
WARMUP_SEC = 30.0


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    config.log_config(recipe_name="lora_finetune_single_device", cfg=cfg)

    recipe = LoRAFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)

    trainable = sum(p.numel() for p in recipe._model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in recipe._model.parameters())
    print(
        f"\n=== Trainable parameters ===\n"
        f"trainable : {trainable:,} ({trainable / 1e6:.2f}M)\n"
        f"total     : {total:,} ({total / 1e6:.2f}M)\n"
        f"fraction  : {100 * trainable / total:.3f}%"
    )

    samples = []  # (elapsed_s_at_step_end, num_tokens_in_step)
    wall_t0 = time.perf_counter()
    stop = False

    def timed_train(self):
        nonlocal stop
        running_loss = 0
        num_tokens = 0
        with self._profiler as prof:
            for curr_epoch in range(self.epochs_run, self.total_epochs):
                self._dataloader.sampler.set_epoch(curr_epoch)
                for idx, batch in enumerate(self._dataloader):
                    utils.batch_to_device(batch, self._device)
                    current_num_tokens = (
                        batch["labels"] != self._loss_fn.ignore_index
                    ).sum()
                    num_tokens += current_num_tokens

                    current_loss = self._loss_step(batch) * current_num_tokens
                    running_loss += current_loss
                    current_loss.backward()

                    if (idx + 1) % self._gradient_accumulation_steps == 0:
                        training.scale_grads(self._model, 1 / num_tokens)
                        if self._clip_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)
                        self._lr_scheduler.step()
                        self.global_step += 1

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        now = time.perf_counter()
                        samples.append((now - wall_t0, int(num_tokens.item())))

                        running_loss = 0
                        num_tokens = 0

                        if now - wall_t0 >= WINDOW_SEC:
                            stop = True
                            return
                    prof.step()
                if stop:
                    return

    recipe.train = types.MethodType(timed_train, recipe)
    recipe.train()
    recipe.cleanup()

    kept = [(t, n) for (t, n) in samples if t >= WARMUP_SEC]
    if len(kept) < 2:
        print(
            f"ERROR: only {len(kept)} post-warmup step(s); "
            "lower batch_size / max_seq_len or raise WINDOW_SEC.",
            file=sys.stderr,
        )
        sys.exit(1)

    t_start, t_end = kept[0][0], kept[-1][0]
    tokens = sum(n for _, n in kept)
    tps = tokens / (t_end - t_start)
    avg_tokens_per_batch = tokens / len(kept)
    print("\n=== Throughput (post-warmup) ===")
    print(f"steps kept           : {len(kept)}")
    print(
        f"window               : {t_end - t_start:.2f} s "
        f"(from {t_start:.2f}s to {t_end:.2f}s of {WINDOW_SEC:.0f}s cap)"
    )
    print(f"tokens processed     : {tokens}")
    print(f"avg tokens / batch   : {avg_tokens_per_batch:.1f}")
    print(f"tokens / second      : {tps:.1f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python run_timed.py llama3_8B_lora.yaml", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
