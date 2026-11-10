# fewshot-glyph-gan

**Teach a tiny, fixed-capacity GAN a brand-new handwritten symbol from 10 drawings.**

A 1.5M-parameter MLP generator (architecture imposed, never modified) is pretrained as a WGAN-GP on 964 Omniglot
characters, then adapted to each *unseen* character from only 10 examples. Adapted generators produce drawings that are
**as novel as fresh human drawings while staying recognisable**, a regime where the strongest hand-tuned augmentation
baseline cannot follow.

<!-- RESULTS_TABLE -->

![fidelity-novelty frontier](docs/figures/frontier.png)

## Where this comes from

The starting point was a course project (kept in [`mnist/`](mnist/)): improve an MNIST generator **without touching its
architecture** (MLP 100-256-512-1024-784). Within that fixed budget we moved from a vanilla GAN to a WGAN, selected models
by validation FID, and added *critic-guided latent refinement* at inference time. Working under a hard capacity
constraint raised a product question: if a small generator is cheap to store and to run, can it be **personalised to a
new symbol** (a logo, a rare glyph, a user's own handwriting) from a handful of examples, the way LoRA personalises large
models?

## Problem and protocol

- **Data.** Omniglot at 28×28, glyph cropped on its ink bounding box and centred (MNIST-style). *Background* alphabets
  (30 alphabets, 964 characters, 19,280 images) are used for pretraining only. *Evaluation* alphabets (20 alphabets,
  659 characters) are never seen before adaptation and are split by alphabet into **dev** (345 characters, all tuning)
  and **test** (314 characters, final numbers only).
- **Episode.** For each new character, its 20 drawings are split into 10 *support* images (the only ones the method sees)
  and 10 *held-out* images (metrics only).
- **Metrics**, in the feature space of a classifier trained on background characters, averaged over characters with
  95% t-intervals:
  - *fidelity*: precision and coverage of the generated samples against the held-out drawings (Kynkäänniemi 2019,
    Naeem 2020); their harmonic mean is the **objective**;
  - *identity*: is a sample closest to the right character among all characters of its alphabet;
  - *novelty*: **feature copy ratio**, the nearest-neighbour distance from generated samples to the support divided by
    the leave-one-out distance within the support. Fresh human drawings score ≈ 1.0; copies score 0.
- **Why novelty must be measured.** With 10 real drawings in hand, simply *returning them* already scores 0.85 on the
  objective (real-data reference 0.89): distribution metrics reward copying. Generation is only useful if it adds new,
  faithful drawings, so methods are compared on the **fidelity–novelty frontier**, and the headline number uses a rule
  fixed *before* looking at test: **maximise the dev objective subject to feature copy ratio ≥ 0.9**, the same rule for
  every method, then replay the selected configuration once on test.

## Method

1. **Pretraining** (`scripts/pretrain.py`). WGAN-GP (λ = 10) with a CNN critic (only the generator is constrained),
   drift penalty, EMA of the generator weights; checkpoint selected by background FID.
2. **Few-shot adaptation** (`scripts/adapt.py`). For each new character, WGAN-GP between its 10 support images and the
   generator, starting from the pretrained generator *and* critic:
   - **generator side**: full fine-tuning, or **LoRA** adapters `W + (α/r)·B·A` (B = 0 at init, so adaptation starts
     exactly at the prior). Merged afterwards, so the deployed generator keeps the original architecture; only `A, B`
     are stored per symbol;
   - **critic side**: DiffAugment translation on real and fake (Zhao 2020) and FreezeD (Mo 2020, lower conv blocks
     frozen). Without DiffAugment the critic memorises the 10 images (its score gap support vs held-out grows 3–4×) and
     the generator follows it into copying.
3. **Inference-time critic refinement** (inherited from the MNIST project): each latent is moved along the normalised
   gradient of the *adapted* critic, `z ← z + η·∇z C(G(z))/‖∇z C(G(z))‖`. No weight changes; one extra point on the
   frontier.
4. **Batched adaptation** (`glyphgan/batched.py`). K characters are adapted *simultaneously* on one GPU, as one
   computation: shared frozen base weights (one matmul), per-character LoRA factors (bmm), K critic copies as grouped
   convolutions, with frozen critic layers computed once for all K. The loss is the sum of the K per-character losses,
   so each character gets exactly its own gradient (Adam is element-wise): mathematically identical to K separate runs,
   verified in `tests/test_batched.py`. 345 characters × 3000 steps take about 25 minutes on one 20 GB GPU (about 4 s
   per character).

## Results

<!-- RESULTS_DETAIL -->

## Baselines and ablations

<!-- ABLATIONS -->

## Limitations

<!-- LIMITATIONS -->

## Reproduce

```bash
uv sync
python -m glyphgan.data --root data                       # Omniglot -> data/omniglot28c.pt
python -m scripts.train_features                          # feature extractor for the metrics
python -m scripts.pretrain --run gp_b --steps 200000 --lr_g 2e-4 --lr_c 2e-4 --beta1 0.5
python -m scripts.select_pretrain runs/pretrain/gp_b
python -m scripts.adapt --ckpt runs/pretrain/gp_b/ckpt_200000.pt --mode full --lr_g 1e-4 --chunk 173 \
       --part dev --steps 3000 --eval_steps 250,500,1000,1500,2000,3000 --run g2_full_lr1e-4
python -m scripts.eval_pools runs/adapt/g2_full_lr1e-4
python -m scripts.select_rule runs/adapt/g2_* --out runs/select_dev.json
```

## Repository

| Path | Content |
|---|---|
| `glyphgan/data.py` | Omniglot preprocessing, dev/test alphabet split, episodes |
| `glyphgan/models.py` | fixed generator, CNN critic, LoRA |
| `glyphgan/batched.py` | K adaptations as one batched computation |
| `glyphgan/features.py`, `metrics.py`, `evaluate.py` | feature extractor, PRDC / copy / identity metrics, evaluation harness |
| `glyphgan/augment.py` | augmentation baseline (affine, elastic, stroke thickness) |
| `scripts/` | pretraining, adaptation, evaluation, baselines, selection rule, figures |
| `mnist/` | the original fixed-architecture MNIST project |
