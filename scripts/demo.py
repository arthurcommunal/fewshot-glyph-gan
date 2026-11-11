"""Interactive demo: draw a new symbol n times, adapt the pretrained generator to it, sample new drawings.

    uv sync --extra demo
    uv run python -m scripts.demo                      # http://127.0.0.1:7860

Each drawing goes through glyphgan.drawing (same 28x28 preprocessing as the training data). "Generate" runs the
configuration selected on dev (LoRA r = 16, 1500 steps, glyphgan.fewshot) on Apple MPS / CUDA if available, else on
CPU with --threads threads (about 20 s on an M-series GPU, about 3 min on 2 CPU threads).

The drawing pad is a plain HTML canvas (gr.Sketchpad took ~3.5 s to export each drawing): "Add" reads it as a PNG
data URL in the browser and clears it in the same click.
"""
import argparse
import base64
import io

import gradio as gr
import numpy as np
import torch
import torchvision
from PIL import Image

from glyphgan.data import to_unit
from glyphgan.drawing import canvas_to_px
from glyphgan.fewshot import Config, adapt_symbol, load_pretrained, sample
from glyphgan.models import Z_DIM

TUNED_N = 10
UPSCALE = 4
PAD = 420
BRUSH = 14

PAD_HTML = f'<canvas id="pad" width="{PAD}" height="{PAD}"></canvas>'
HEAD = """<style>
#pad { background: #fff; border-radius: 8px; touch-action: none; cursor: crosshair; display: block; margin: 0 auto;
       width: 100%; max-width: %PAD%px; aspect-ratio: 1; }
</style>
<script>
(() => {
  let drawing = false, last = null;
  const pos = (c, e) => { const r = c.getBoundingClientRect();
    return [(e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height]; };
  const line = (c, a, b) => { const g = c.getContext("2d");
    g.strokeStyle = "#000"; g.lineWidth = %BRUSH%; g.lineCap = "round"; g.lineJoin = "round";
    g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke(); };
  document.addEventListener("pointerdown", e => { if (e.target.id !== "pad") return;
    drawing = true; last = pos(e.target, e); line(e.target, last, last); e.target.setPointerCapture(e.pointerId); });
  document.addEventListener("pointermove", e => { if (!drawing || e.target.id !== "pad") return;
    const p = pos(e.target, e); line(e.target, last, p); last = p; });
  document.addEventListener("pointerup", () => { drawing = false; });
  window.padClear = () => { const c = document.getElementById("pad"); c.getContext("2d").clearRect(0, 0, c.width, c.height); };
  window.padTake = () => { const c = document.getElementById("pad"); const url = c.toDataURL("image/png"); window.padClear(); return url; };
})();
</script>""".replace("%PAD%", str(PAD)).replace("%BRUSH%", str(BRUSH))


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode(data_url: str) -> np.ndarray:
    """PNG data URL (transparent background, black strokes) -> (H, W, 4) uint8."""
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGBA"))


def grid(x: torch.Tensor, nrow: int = 8) -> Image.Image:
    """x (N, 784) in [-1, 1] -> PIL grid, white strokes on black, upscaled for display."""
    g = torchvision.utils.make_grid((x.view(-1, 1, 28, 28).cpu() + 1) / 2, nrow=nrow, padding=2, pad_value=0.25)
    img = Image.fromarray((g[0].clamp(0, 1).numpy() * 255).astype(np.uint8))
    return img.resize((img.width * UPSCALE, img.height * UPSCALE), Image.NEAREST)


def strip(drawings: list) -> Image.Image | None:
    """The collected drawings as the model sees them (28x28 each), 5 per row."""
    return grid(to_unit(torch.from_numpy(np.stack(drawings))), nrow=5) if drawings else None


def status(n: int) -> str:
    if n == 0:
        return "No drawing yet. Draw the same symbol several times, clicking **Add** after each one."
    note = "" if n == TUNED_N else f" (the method was tuned with {TUNED_N})"
    return f"**n = {n}** drawing{'s' if n > 1 else ''}{note}."


def build(G, C, dev: torch.device, cfg: Config, n_samples: int) -> gr.Blocks:
    z_prior = torch.randn(n_samples, Z_DIM, generator=torch.Generator().manual_seed(1))

    def add(data_url, drawings):
        px = canvas_to_px(decode(data_url)) if data_url else None
        if px is None or px.max() == 0:
            raise gr.Error("The canvas is empty.")
        drawings = drawings + [px]
        return drawings, strip(drawings), status(len(drawings))

    def undo(drawings):
        drawings = drawings[:-1]
        return drawings, strip(drawings), status(len(drawings))

    def reset():
        return [], None, status(0), None, None

    def generate(drawings, progress=gr.Progress()):
        if not drawings:
            raise gr.Error("Add at least one drawing first.")
        support = to_unit(torch.from_numpy(np.stack(drawings))).to(dev)
        with torch.no_grad():
            prior = G.cpu()(z_prior)

        def cb(step, total):
            if step % 25 == 0 or step == total:
                progress(step / total, desc=f"Adapting the generator: step {step}/{total}")

        progress(0, desc="Adapting the generator")
        Gb, Cb = adapt_symbol(G.to(dev), C.to(dev), support, cfg, callback=cb)
        progress(1, desc="Sampling")
        x = sample(Gb, Cb, n_samples, cfg)
        return grid(prior), grid(x)

    with gr.Blocks(title="Few-shot glyph GAN") as demo:
        gr.Markdown(
            "## Few-shot glyph GAN\n"
            "Draw a new symbol a few times. **Generate** adapts a fixed 1.5 M-parameter MLP generator, pretrained on "
            "Omniglot, to your symbol (LoRA, 1500 steps), then samples new drawings of it.")
        drawings = gr.State([])
        data_url = gr.Textbox(visible=False)
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(PAD_HTML)
                with gr.Row():
                    add_btn = gr.Button("Add", variant="secondary")
                    clear_btn = gr.Button("Clear canvas")
                with gr.Row():
                    undo_btn = gr.Button("Undo last drawing")
                    reset_btn = gr.Button("Reset")
            with gr.Column(scale=1):
                info = gr.Markdown(status(0))
                gallery = gr.Image(label="Your drawings, as the model sees them (28x28)", type="pil", height=300)
                gen_btn = gr.Button("Generate", variant="primary")
        with gr.Row():
            before = gr.Image(label="Pretrained generator (before adaptation)", type="pil")
            after = gr.Image(label="Adapted to your symbol", type="pil")

        add_btn.click(add, [data_url, drawings], [drawings, gallery, info], js="(u, d) => [window.padTake(), d]")
        clear_btn.click(None, js="() => window.padClear()")
        undo_btn.click(undo, [drawings], [drawings, gallery, info])
        reset_btn.click(reset, None, [drawings, gallery, info, before, after], js="() => window.padClear()")
        gen_btn.click(generate, [drawings], [before, after])
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/gp_b_200k.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=2, help="CPU threads used by torch")
    p.add_argument("--steps", type=int, default=Config.steps)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    dev = pick_device(a.device)
    G, C = load_pretrained(a.ckpt)
    print(f"device: {dev}", flush=True)
    demo = build(G, C, dev, Config(steps=a.steps), a.samples)
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=a.port, show_error=True,
                                                   head=HEAD)


if __name__ == "__main__":
    main()
