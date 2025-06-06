### Install
Create a new Conda environment:
```bash
conda create -n vtoff python=3.11
conda activate vtoff
```

Then, clone the repository, switch to commit used for the paper, install the required packages:
```bash
git clone https://github.com/rizavelioglu/tryoffdiff.git
git switch c385ea2
cd tryoffdiff
pip install -e .
```

### Dataset
Download the original VITON-HD dataset and extract it to `"./data/vitonhd"`:
```bash
python tryoffdiff/dataset.py download-vitonhd  # For a different location: output-dir="<other-folder>"
```
As mentioned in the paper, the original dataset contains duplicates, and some training samples are leaked into the test
set. Clean these with the following command:
```bash
python tryoffdiff/dataset.py clean-vitonhd  # Default: `data-dir="./data/vitonhd"`
```

### Training
For faster training, pre-extract the image features and save them instead of extracting them during training.

#### Step 1: Encode garment images with VAE
```bash
python tryoffdiff/dataset.py vae-encode-vitonhd \
 --data-dir "./data/vitonhd/" \
 --model-name "sd14" \
 --batch-size 16
 ```

#### Step 2: Encode model(conditioning) images with SigLIP
```bash
python tryoffdiff/dataset.py siglip-encode-vitonhd \
 --data-dir "./data/vitonhd/" \
 --batch-size 64
 ```

#### Step 3: Train `TryOffDiff`

- Option 1 (GPU-poor) - Train with a single GPU:

Execute the following
```bash
python tryoffdiff/modeling/train.py tryoffdiff \
 --save-dir "./models/" \
 --data-dir "./data/vitonhd-enc-sd14/" \
 --model-class-name "TryOffDiff" \
 --mixed-precision "no" \
 --learning-rate 0.0001 \
 --train-batch-size 16 \
 --num-epochs 1200 \
 --save-model-epochs 100 \
 --checkpoint-every-n-epochs 100
```

- Option 2 - Train with 4-GPUs on a single node (as done in the paper):

First, configure `accelerate` accordingly:
```bash
accelerate config
```
> We did not use any of the tools like dynamo, DeepSpeed, FullyShardedDataParallel etc.

Then, start training:
```bash
accelerate launch --multi_gpu --num_processes=4 tryoffdiff/modeling/train.py tryoffdiff \
 --save-dir "./models/" \
 --data-dir "./data/vitonhd-enc-sd14/" \
 --model-class-name "TryOffDiff" \
 --mixed-precision "no" \
 --learning-rate 0.0001 \
 --train-batch-size 16 \
 --num-epochs 1201 \
 --save-model-epochs 100 \
 --checkpoint-every-n-epochs 100
```

> **Note**: See [config.py(TrainingConfig)](tryoffdiff/config.py) for all possible arguments, e.g. set `resume_from_checkpoint` to resume
training from a specific checkpoint.


### Inference
Each model has its own command. View all available options:
```bash
python tryoffdiff/modeling/predict.py --help
```

> Example: Run inference with `TryOffDiff`:
> ```bash
> python tryoffdiff/modeling/predict.py tryoffdiff \
>  --model-dir "/model_20241007_154516/" \
>  --model-filename "model_epoch_1200.pth" \
>  --batch-size 8 \
>  --num-inference-steps 50 \
>  --seed 42 \
>  --guidance-scale 2.0
> ```
which saves predictions to `"<model-dir>/preds/"` as `.png` files.

> **Note**: See [config.py(InferenceConfig)](tryoffdiff/config.py) for all possible arguments,
e.g. use the `--all` flag to run inference on the entire test set.

> **Note**: The paper uses the PNDM noise scheduler. For HuggingFace Spaces we use the EulerDiscrete scheduler.

### Evaluation

Evaluate the predictions using:
```bash
python tryoffdiff/modeling/eval.py \
 --gt-dir "./data/vitonhd/test/cloth/" \
 --pred-dir "<prediction-dir>" \
 --batch-size 32 \
 --num-workers 4
 ```
which prints the results to the console.
Specifically, we use the following libraries for the implementations of the metrics presented in the paper:
- `pyiqa`: `SSIM`, `MS-SSIM`, `CW-SSIM`, and `LPIPS`,
- `clean-fid`: `FID`, `CLIP-FID`, and `KID`,
- `DISTS-pytorch`: `DISTS`

In addition, we offer a simple GUI for visualizing predictions alongside their evaluation metrics. This tool displays the ground truth and predicted images side-by-side while providing metrics for the entire test set:
```bash
python tryoffdiff/modeling/eval_vis.py \
 --gt-dir "./data/vitonhd/test/cloth/" \
 --pred-dir "<prediction-dir>"
```

---
<div style="display: flex; justify-content: space-between;">

   [Back](README.md)

   [Next: v2-instructions](v2_instructions.md)

</div>
