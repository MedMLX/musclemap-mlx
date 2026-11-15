# MuscleMap v1.4 implementation and qualification

## Reference contract

The source reference is [MuscleMap/MuscleMap at
35a2138](https://github.com/MuscleMap/MuscleMap/tree/35a2138a4a3a63be713bae5584632ffc6433b960),
with the contrast-agnostic wholebody v1.4 checkpoint and JSON configuration from
[Zenodo record 21929873](https://zenodo.org/records/21929873). Both source and
publisher assets carry the MIT grant. Attribution and the complete license travel
with the converted bundle. Conversion uses `torch.load(weights_only=True)` and
the standard safetensors writer, followed by exact tensor comparisons.

The network is MONAI's 2D residual UNet: one input, 114 outputs, channels
(64, 128, 256, 512, 1024), strides (2, 2, 2, 2), two residual units, instance
normalization with epsilon 1e-5 and no affine parameters, LeakyReLU slope 0.01,
and the upstream residual and transposed-convolution ordering. All 46 float32
checkpoint tensors retain their original names and values. The complete publisher
label table is checked against the runtime: all 113 IDs, names, sides, regions and
channel order must match. Background is channel/ID 0; anatomical IDs are not
renumbered in the saved mask.

Preprocessing follows `mm_segment.py`: apply NIfTI scaling, orient to RAS,
resample to `[1, 1, -1]` mm (retain source slice spacing), normalize nonzero
voxels, crop positive foreground with margin 20, then pad at the end to at least
256x256x1. The slice inferer uses 256x256 windows, batch size 1, Gaussian sigma
scale 0.125 and 90% overlap. Inverse pad/crop, bilinear spacing restoration and
orientation run on logits before argmax and anatomical remapping. Largest
6-connected components are selected after assembling all chunks.

Chunking follows `mm_util.run_inference`: slice the last source storage axis,
write each float32 chunk using the source NIfTI header, then preprocess each
chunk independently. NiBabel rescales float32 data stored with an integer header;
this can change zeros and therefore nonzero normalization. The native runtime
reproduces that round trip using a private temporary directory. Skipping it
initially caused 228 CT and 190 MRI final-voxel differences in these examples.
The default is eight source slices; a chunk size at least the depth skips chunk
serialization and processes the whole image. Comparisons require the same
chunk size.

Network operations run through MLX on Metal in float32. NIfTI I/O, geometry,
Gaussian accumulation, argmax and connected components run on the host. An
unavailable Metal device fails explicitly. Stored output is int16, with the
original affine, distinct qform/sform matrices and codes, and unit metadata.
Unknown spatial units are interpreted as mm while retaining the original unit
field; explicitly different units are rejected.

## Real inputs and acceptance criteria

The complete compact CT and MRI test volumes are from
[TotalSegmentator at d89e47f](https://github.com/wasserth/TotalSegmentator/tree/d89e47f73d24f5fba732a715aeb6fdc0ae34526b/tests/reference_files).
Its [official test commands](https://github.com/wasserth/TotalSegmentator/blob/d89e47f73d24f5fba732a715aeb6fdc0ae34526b/tests/tests.sh)
supply the CT input to CT inference and the MRI input to `total_mr`; modality
comes from that source definition. These are real
images supplied as compact test examples, rather than newly synthesized images.
Their NIfTI affines define 3 mm voxels; both have unspecified unit fields, qform
code 0 and sform code 2. No header fields were altered for the experiment.

| Input | Shape | Orientation | Foreground IDs present |
| --- | --- | --- | --- |
| `example_ct_sm.nii.gz` | 122x101x30 | RAS | 10 |
| `example_mr_sm.nii.gz` | 117x91x20 | LPS | 10 |

Acceptance was fixed before comparison: finite float32 network and restored
logits within `atol=5e-4, rtol=1e-5`; exact final anatomical masks after component
filtering; equal output shape, affine, spatial forms/codes and units; int16
storage. The same tensor windows are used for network comparisons. Restored-logit
comparisons include native preprocessing, Gaussian aggregation and inverse
geometry. CPU reduction rounding is reported separately.

## Results

Qualification on 2026-11-13 used Python 3.13, MLX 0.32.4 and float32 on an Apple
Silicon Mac. The independent reference used Python 3.11,
PyTorch 2.4.1, MONAI 1.3.2, NumPy 1.24.4, NiBabel 5.2.1 and SciPy 1.11.4,
with one CPU thread. Reference execution completed both on the Mac and on a
DGX Spark GB10 using its CPU with the same pinned dependencies. Both examples
used eight-slice chunks and 90% overlap. The table reports the largest error
across these two independent reference executions.

| Measurement | CT | MRI |
| --- | --- | --- |
| Maximum network-logit absolute error | 1.47e-4 | 1.36e-4 |
| Maximum restored-logit absolute error | 1.15e-4 | 1.03e-4 |
| Final mismatched voxels | 0 / 369,660 | 0 / 212,940 |
| Dice for every foreground ID present | 1.0 | 1.0 |
| Output grid, forms and units | Preserved | Preserved |

CT preprocessing differed by at most 4.77e-7 from the reference after float32
normalization; MRI preprocessing matched exactly. Reference reader pixels were
checked independently with pinned NiBabel. The complete public package runner
also produced the exact reference masks. These measurements establish parity on
two compact examples, without measuring manual-label accuracy or every one of
the 113 anatomical structures.

## Inference performance

The optimized runtime compiles the float32 network, exports contiguous
channel-first logits, and blends independent slices as a 2D image batch. The
next Metal prediction overlaps host blending. Inverse resampling writes directly
to float32 output and processes up to eight independent channels concurrently;
SciPy retains double-precision interpolation arithmetic. Window batch size,
overlap, chunking, normalization and label order are preserved.

Measurements on 2026-11-14 used an Apple M5 Max with 128 GB memory, macOS 27.2,
Python 3.13.17, MLX 0.32.4, SciPy 1.18.1 and full float32 products. The baseline
was the inference code at `d1f5538`. Each of six alternating process pairs ran
one cold and one warmed segmentation of the complete compact images above,
with eight-slice chunks and 90% overlap. Timing includes preprocessing, chunk
serialization, network execution, blending, inverse geometry and component
filtering; initial image loading and final mask writing are outside the timer.

| Input | Warm baseline | Warm optimized | Cold baseline | Cold optimized | Warm speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| CT | 16.43 s | 6.81 s | 16.12 s | 6.82 s | 2.41x |
| MRI | 6.02 s | 2.48 s | 6.22 s | 2.44 s | 2.43x |

Times are medians across the six pairs. The paired 97% intervals for the
optimized/baseline warm-time ratio were 0.329–0.480 for CT and 0.373–0.448 for
MRI. Peak process-footprint increases over setup were 1839→1860 MB for CT and
1764→1784 MB for MRI, within the 2% comparison threshold. These process counters
include host arrays and are distinct from the MLX allocator counter below.

Both complete masks matched the recorded MONAI reference exactly (582,600
voxels total). All seven recorded network windows passed the existing
`atol=5e-4, rtol=1e-5` contract, with maximum absolute error 1.42e-4. The layout
and scheduling comparison also preserved all 24,980,592 blended logits in a
two-slice CT workload bit for bit.

Chunked runs then prepare the next chunk (source-header round trip and
preprocessing) on one worker thread while the current chunk runs on Metal. On
2026-11-15, with the same setup and twelve alternating pairs against `7a5169f`,
warm CT time fell from 5.79 s to 5.50 s (paired 96% ratio interval
0.914–0.956); MRI fell from 2.37 s to 2.26 s (0.938–0.996). Masks were
identical and peak process footprint was unchanged. A window runs in about
6.9 ms, which is now about 85% of chunk time. Zero-padding the 114-channel top
layer to 128 channels reduced it to 6.3 ms but selected an MLX convolution with
about 167 MB more peak memory, so it was not adopted.

## Peak memory

The README's peak MLX memory is `details.peak_memory_bytes` (`mx.get_peak_memory()`
after `mx.reset_peak_memory()`) from one `musclemap-mlx==0.1.1` run with default
settings (eight-slice chunks, 90% overlap) on a 543x543x568 autoPET-III whole-body CT
resampled to 1.5 mm: 542,336,612 bytes, measured on 2026-11-13 with MLX 0.32.4.
A repeat `musclemap-mlx==0.1.1` run on the same CT on 2026-11-14 (MLX 0.32.4) reported the
same peak and `details.duration_seconds` of 1309 s.

## Reproduce

Install the published runtime and download the public qualification tools at their
reviewed revision. The reference uses Python 3.11; native execution uses Python
3.13 on Apple Silicon. The source distribution supplies the configuration parser
for the reference environment, without installing an MLX runtime there.

```sh
uv venv --python 3.13
source .venv/bin/activate
uv pip install musclemap-mlx==0.2.0
hf download MedMLX/musclemap-mlx --revision 05fa11f1f13ded18d2efdb7a8e1dab77bd600234 --include 'qualification/*' --local-dir validation

# Obtain the same published source used by the installed runtime.
python - <<'PYCODE'
import json, urllib.request
metadata = json.load(urllib.request.urlopen("https://pypi.org/pypi/musclemap-mlx/0.2.0/json"))
url = next(item["url"] for item in metadata["urls"] if item["packagetype"] == "sdist")
urllib.request.urlretrieve(url, "musclemap_mlx-0.2.0.tar.gz")
PYCODE
tar -xf musclemap_mlx-0.2.0.tar.gz
uv venv --python 3.11 oracle-env
uv pip install --python oracle-env/bin/python -r validation/qualification/requirements.txt
mkdir -p weights results
curl -L https://zenodo.org/records/21929873/files/contrast_agnostic_wholebody_model.pth -o weights/contrast_agnostic_wholebody_model.pth
curl -L https://zenodo.org/records/21929873/files/contrast_agnostic_wholebody_model.json -o weights/contrast_agnostic_wholebody_model.json
curl -L https://zenodo.org/records/21929873/files/LICENSE -o weights/LICENSE
curl -L https://raw.githubusercontent.com/wasserth/TotalSegmentator/d89e47f73d24f5fba732a715aeb6fdc0ae34526b/tests/reference_files/example_ct_sm.nii.gz -o ct.nii.gz
curl -L https://raw.githubusercontent.com/wasserth/TotalSegmentator/d89e47f73d24f5fba732a715aeb6fdc0ae34526b/tests/reference_files/example_mr_sm.nii.gz -o mr.nii.gz

PYTHONPATH=musclemap_mlx-0.2.0/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  oracle-env/bin/python validation/qualification/make_reference_fixtures.py \
  --assets weights --image ct.nii.gz --output results/ct-reference
musclemap-mlx download --model-dir converted
python validation/qualification/qualify_real.py \
  --reference results/ct-reference --weights converted --output results/ct-parity.json
```

Repeat the last reference and comparison commands with `mr.nii.gz` and separate
MRI result paths. The recorder can run on Linux; copy its result folder to the
Mac for native comparison. `qualify_real.py` exits unsuccessfully for logit
failures or any final mask mismatch and records per-chunk errors and per-label
Dice. Generated images, weights, reference arrays and run output stay outside Git.
