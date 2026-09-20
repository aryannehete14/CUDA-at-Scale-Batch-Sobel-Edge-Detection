# CUDA at Scale — Batch Sobel Edge Detection

## Overview
This project applies **Sobel edge detection** to a large batch of grayscale
images entirely on the GPU using a custom CUDA kernel. It satisfies the
"many small inputs" variant of the assignment: it is designed to run on
**hundreds of images in a single execution**, timing each one individually
and in aggregate.

Each image is:
1. Loaded from disk (binary PGM format).
2. Copied to GPU memory.
3. Convolved with the Sobel Gx and Gy operators (3x3 kernels, stored in
   `__constant__` memory) to compute a gradient-magnitude edge map, one
   CUDA thread per output pixel.
4. Copied back to host memory and written to disk.

## Why Sobel / why this design
Sobel edge detection is a classic, easily-verified image processing
operator — the output is visually self-explanatory (edges become bright,
flat regions become dark), which makes correctness easy to confirm just by
looking at before/after images. It's also a good demonstration of
data-parallel GPU work: each output pixel is computed independently, with
no cross-thread synchronization needed, making it a natural fit for CUDA's
thread-per-pixel model.

## Repository layout
```
.
├── src/
│   ├── main.cu            # CLI, batch loop, timing, CUDA memory management
│   ├── sobel_kernel.cu    # The CUDA kernel itself
│   ├── sobel_kernel.cuh   # Kernel launch declaration
│   └── pgm_io.h           # Minimal PGM (P5) reader/writer, no dependencies
├── generate_data.py       # Generates synthetic sample images (no deps)
├── Makefile                # nvcc build
├── run.sh                  # Build + generate data + run, one command
├── data/                   # Input images (generated or your own)
├── output/                 # Edge-detected output images
└── logs/                   # Per-run proof-of-execution logs
```

## Requirements
- NVIDIA GPU + CUDA Toolkit (`nvcc` on PATH)
- Python 3 (only for the optional synthetic data generator; standard
  library only, no PIL/numpy needed)

This project was developed and run on **Google Colab** (free tier, T4 GPU),
since a local NVIDIA GPU was not available. See the Colab instructions
below if you're in the same situation.

## Build and run on Google Colab
Newer Colab runtimes ship the NVIDIA driver but not always the CUDA
compiler by default, so install `nvcc` first:

```python
!pip install -q nvidia-cuda-nvcc-cu12
import os
os.environ['PATH'] = '/usr/local/lib/python3.13/dist-packages/nvidia/cuda_nvcc/bin:' + os.environ['PATH']
!nvcc --version
```

(Adjust the `python3.13` path segment to match whatever `!which python3`
reports if it differs.)

Then, with this repo cloned or uploaded into the Colab filesystem:

```python
!python3 generate_data.py --count 200 --size 256 --out data
!nvcc -O3 -std=c++14 -arch=sm_75 src/main.cu src/sobel_kernel.cu -o bin_edge_detect
!mkdir -p output logs
!./bin_edge_detect --input data --output output --log logs/run.log
!cat logs/run.log
```

`-arch=sm_75` matches Colab's free-tier T4 GPU. If Colab assigns you a
different GPU, check its compute capability with `!nvidia-smi` and adjust
accordingly (A100 → `sm_80`, L4 → `sm_89`, etc.).

## Build and run on a machine with a local GPU + CUDA Toolkit
```bash
make
```
This produces `bin/edge_detect`. Adjust `-arch=sm_50` in the `Makefile`
if targeting a different GPU compute capability.

### Option 1: one-command build + sample data + run
```bash
./run.sh            # generates 200 synthetic 256x256 images and processes them
./run.sh 500 128     # generates 500 synthetic 128x128 images instead
```

### Option 2: manual steps
```bash
python3 generate_data.py --count 200 --size 256 --out data/
make
./bin/edge_detect --input data --output output --log logs/run.log
```

### Option 3: use your own dataset
Convert any image set to binary PGM (e.g. with ImageMagick:
`convert input.jpg -colorspace Gray output.pgm`) and place the `.pgm`
files in a directory, then:
```bash
./bin/edge_detect --input /path/to/your/pgms --output output --log logs/run.log
```

## CLI arguments
| Flag       | Required | Description                              |
|------------|----------|-------------------------------------------|
| `--input`  | yes      | Directory of `.pgm` images to process     |
| `--output` | yes      | Directory to write edge-detected images   |
| `--log`    | no       | Path to write the run log (default shown) |

## Proof of execution
Running the build/run steps above produces:
- `logs/run.log` — lists every image processed, its dimensions, per-image
  GPU kernel time in milliseconds, and aggregate kernel/wall-clock time
  across the whole batch — clear evidence the kernel executed on many
  images within a single program run, not just once.
- `output/*.pgm` — the edge-detected images themselves, viewable with any
  PGM-capable viewer (e.g. GIMP, IrfanView), in a Colab notebook with
  `PIL.Image.open(...)`, or converted with
  `convert output/image_0000.pgm output/image_0000.png`.

## Lessons learned / notes
- No local NVIDIA GPU was available for development, so the whole
  build/run/test loop was done on Google Colab's free-tier GPU runtime.
  The main practical wrinkle was that `nvcc` isn't preinstalled on newer
  Colab images by default and had to be added via
  `pip install nvidia-cuda-nvcc-cu12`.
- Boundary handling (pixels at image edges) is done via clamped indexing
  inside the kernel rather than padding the image on the host — simpler
  memory management at a small cost in warp divergence at the border.
- Sobel's constant 3x3 kernels are placed in `__constant__` memory since
  they're small, read-only, and identical for every thread — this lets
  the GPU broadcast them efficiently instead of re-reading from global
  memory per thread.
- The current version launches one kernel per image using a single CUDA
  stream. A natural extension (not required for this assignment) would
  be to use multiple streams to overlap host-to-device transfer of image
  *N+1* with kernel execution on image *N*.
- No external image libraries are used (no OpenCV/stb) to keep the build
  dependency-free in constrained lab environments; PGM was chosen as a
  minimal, trivially-parseable format for that reason.

## Author
[Your name here]