### Install
Create a new Conda environment:
```bash
conda create -n vtoff python=3.11
conda activate vtoff
```

Then, clone the repository, switch to commit used for the paper, install the required packages:
```bash
git clone https://github.com/rizavelioglu/tryoffdiff.git
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

Download the Dress Code dataset ([follow steps here](https://github.com/aimagelab/dress-code)) and execute the following command, to match the folder structure of VITON-HD:
```bash
python tryoffdiff/dataset.py restructure_dresscode_to_vitonhd --zip-dir "/path/to/DressCode.zip"
```
which takes the downloaded Dress Code zip file, extracts only required folders, and restructures it to match the VITON-HD folder structure:
```
dresscode/
├── test/
│   ├── cloth/
│   │   ├── 048393_1.jpg
│   │   ├── 048394_1.jpg
│   │   └── ...
│   └── image/
│   │   ├── 048393_0.jpg
│   │   ├── 048394_0.jpg
│       └── ...
└── train/
    ├── cloth/
    │   ├── 000000_1.jpg
    │   ├── 000001_1.jpg
    │   └── ...
    └── image/
        ├── 000000_0.jpg
        ├── 000001_0.jpg
        └── ...
```

### Training
For faster training, pre-extract the image features and save them instead of extracting them during training.

#### Step 1: Encode garment images with VAE
```bash
python tryoffdiff/dataset.py vae-encode-dataset \
 --data-dir "./data/dresscode/" \
 --model-name "sd14" \
 --batch-size 32 \
 --data-name "dresscode"
 ```

> **Note:** We do not store SigLIP features for the Dress Code dataset, as it is 4x larger than VITON-HD and
would require significantly more disk space. Instead, features are extracted on-the-fly during training.

#### Step 2: Train `TryOffDiffv2` models with a single GPU:
To train a model for a specific garment type, such as upper body garments, use the following command:
```bash
python tryoffdiff/modeling/train.py tryoffdiffv2 \
 --save-dir "./models/" \
 --data-dir "./data/dresscode-enc-sd14/" \
 --model-class-name "TryOffDiffv2Single" \
 --mixed-precision "no" \
 --learning-rate 0.0001 \
 --train-batch-size 16 \
 --num-epochs 200 \
 --save-model-epochs 100 \
 --checkpoint-every-n-epochs 100 \
 --dataset-type "dc-upperbody"  # or "dc-lowerbody", "dc-dresses"
```

To train with the entire dataset, supporting all garment types, use the following command:
```bash
python tryoffdiff/modeling/train.py tryoffdiffv2 \
 --save-dir "./models/" \
 --data-dir "./data/dresscode-enc-sd14/" \
 --model-class-name "TryOffDiffv2Multi" \  # Multi-garment TryOffDiff
 --mixed-precision "no" \
 --learning-rate 0.0001 \
 --train-batch-size 16 \
 --num-epochs 200 \
 --save-model-epochs 100 \
 --checkpoint-every-n-epochs 100 \
 --dataset-type "dresscode"  # Entire dataset with all garment types
```


### Inference
To run inference with a trained model (regardless of its model class, i.e. `TryOffDiffv2Single` or `TryOffDiffv2Multi`),
 simply use the following command:

```bash
python tryoffdiff/modeling/predict.py tryoffdiffv2 \
  --model-dir "/model_20250710_123456/" \
  --model-filename "model_epoch_200.pth" \
  --batch-size 8  \
  --num-inference-steps 20 \
  --seed 42 \
  --guidance-scale 2.5
```

which runs inference on the entire test set and  saves predictions to `"<model-dir>/preds/"` as `.png` files.

### Evaluation

Evaluate the predictions using:
```bash
python tryoffdiff/modeling/eval.py \
 --gt-dir "./data/dresscode/test/cloth/" \
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
 --gt-dir "./data/dresscode/test/cloth/" \
 --pred-dir "<prediction-dir>"
```


---
<div style="display: flex; justify-content: space-between;">

   [Back](v1_instructions.md)

   [To main page](../README.md)

</div>
