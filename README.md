# Oasis-Memory: Improving DiT Temporal Consistency with Memory Banks

![Header Animation](https://raw.githubusercontent.com/TheApplePieGod/oasis-improved-memory/main/images/header.gif)

A research project that enhances the [Oasis](https://oasis-model.github.io/) diffusion transformer (DiT) model with a memory bank system to improve temporal consistency in AI-generated Minecraft gameplay.

**Final project for 6.7960, MIT**

**Authors:** [Evan Thompson](https://evanthompson.site), [Lowell Torola](https://github.com/lowtorola), [Hank Stennes](https://github.com/hstennes)

---

## Overview

The Oasis model uses a diffusion transformer to emulate Minecraft in real-time while responding to user input. While good in performance, the model exhibits limited memory over time due to its constrained frame-based history architecture. When the player turns around or looks up, the landscape surrounding them often changes inconsistently.

**Oasis-Memory** addresses this limitation by introducing a medium-term memory pipeline that dramatically improves temporal consistency and reduces failure modes, even with fewer training iterations.

## Key Features

- **Memory Bank System**: Maintains a fixed-length context of memory encodings using a FIFO replacement policy with cosine similarity thresholding
- **Memory Compression**: Drastically downsamples and quantizes frames for efficient long-term storage
- **Improved Temporal Consistency**: Preserves biome information, terrain structure, and environmental details across frames
- **Efficient Architecture**: Uses a simple linear embedder that successfully conditions frame generation on historical context

## Architecture

![Oasis-Memory Architecture](https://raw.githubusercontent.com/TheApplePieGod/oasis-improved-memory/main/images/oasis-memory.jpg)

The system consists of three main components:

1. **Memory Encoder Module**: Processes input frames and generates compressed memory vectors using a VAE-based encoder
2. **Memory Bank**: Stores a fixed-length context of memory encodings
   - FIFO replacement policy with cosine similarity thresholding
   - Only adds entries when environment changes sufficiently
   - Enables memory over much longer context lengths
3. **Memory Embedder**: Converts memory snapshots into embeddings for DiT conditioning using a linear mapping

## Results

Our experiments demonstrate significant improvements over the baseline:

### Walk Scenario
Baseline models quickly lose biome and terrain consistency:

![Walk Comparison](https://raw.githubusercontent.com/TheApplePieGod/oasis-improved-memory/main/images/results/better_baseline_walk.gif)

Oasis-Memory with Linear + MC-Small maintains temporal consistency:

![Improved Walk](https://raw.githubusercontent.com/TheApplePieGod/oasis-improved-memory/main/images/results/dit_linear_mim_10k_walk.gif)

### Sky Scenario
The most challenging test: looking up at the sky, then looking back down should restore the original landscape.

**Baseline** (fails completely):

![Baseline Sky](https://raw.githubusercontent.com/TheApplePieGod/oasis-improved-memory/main/images/results/baseline_sky.gif)

**Oasis-Memory** (successfully recovers landscape from memory):

![Improved Sky](https://raw.githubusercontent.com/TheApplePieGod/oasis-improved-memory/main/images/results/dit_linear_mim_10k_sky_1.gif)

The model correctly adapts to different biomes stored in memory:

![Multiple Biomes](https://raw.githubusercontent.com/TheApplePieGod/oasis-improved-memory/main/images/results/dit_linear_mim_10k_sky_2.gif)

## Training Details

- **VAE Training**: 20k iterations on ~3000 videos from the Minecraft dataset
- **DiT Training**: ~10k iterations on custom labeled dataset of 44 videos (22 biomes)
- **Frame Resolution**: 320x160 (half of original Oasis resolution for faster training)
- **Context Window**: 16 frames (with 16-entry memory bank)
- **VAE Loss Functions**: MSE reconstruction loss, KL-divergence loss, LPIPS loss

## Experimental Scenarios

We designed three benchmarking scenarios:

1. **Walk**: User walks forward through a biome with fixed viewing angle
   - Tests baseline usability and terrain preservation

2. **Walk-Frozen**: Walk scenario with frozen memory bank
   - Tests eviction policy effectiveness
   - Evaluates whether DiT respects memory context

3. **Sky**: Most challenging - looks up at sky, then back down
   - Tests memory recall when looking away and back
   - Should restore original landscape from memory

## Key Findings

- **Memory compression (MC-small) outperforms uncompressed memory (MC-large)**
  - Compressed memory reduces noise and encodes meaningful structures
  - Smaller memory footprint enables longer context

- **Simple linear embedder is more effective than transformer-based MiT**
  - MiT outputs were ignored by DiT during concurrent training
  - Linear embedder provides immediate useful signal

- **Memory bank demonstrably improves temporal consistency even with short training**
  - Walk-frozen experiments prove DiT attends to memory contents
  - Sky scenario shows successful landscape recovery

- **FIFO replacement with cosine similarity effectively manages memory updates**
  - Only adds entries when environment changes significantly
  - Preserves relevant long-term context

## Limitations

### Current Limitations

- Experiments conducted on small overfit dataset (44 videos) due to compute constraints
- DiT occasionally "anticipates" look-down actions in sky scenario
- MiT module was not used effectively (likely due to concurrent training issues)
- Only tested on Minecraft environment

## Background and Related Work

This project builds upon several key papers:

- **Oasis** - Decart, Etched et al. [[Website]](https://oasis-model.github.io/)
  - Base model for real-time Minecraft generation using DiT

- **Diffusion Models are Real-Time Game Engines** - Valevski et al. [[Paper]](https://arxiv.org/pdf/2408.14837)
  - Similar approach for DOOM using diffusion models

- **World Models** - Ha and Schmidhuber [[Paper]](https://arxiv.org/pdf/1803.10122)
  - Inspiration for VAE + RNN memory system architecture

- **Diffusion Forcing** - Chen et al. [[Paper]](https://arxiv.org/abs/2407.01392)
  - Training methodology for diffusion transformers

- **Latent Diffusion Models** - Rombach et al. [[Paper]](https://arxiv.org/abs/2112.10752)
  - VAE scaling factor computation

See the [full blog post](67960Final/index.html) for complete citations and detailed methodology.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{Thompson2024OasisMemory,
  title={Oasis-Memory: Improving DiT Temporal Consistency with Memory Banks},
  author={Thompson, Evan and Torola, Lowell and Stennes, Hank},
  year={2024},
  institution={MIT},
  note={6.7960 Final Project}
}
```