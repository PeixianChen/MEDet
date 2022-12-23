# Open Vocabulary Object Detection with Proposal Mining and Prediction Equalization

### Installation
```bash
conda create --name detic python=3.8 -y
conda activate detic
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

cd ..
pip install -r requirements.txt
```

### Benchmark evaluation and training
- Training
```
python train_net.py --config-file ./configs/MEDet_COCO_CLIP_R50.yaml --num-gpus $GPUS$
```
- Evaluation
```
python train_net.py --config-file ./configs/MEDet_COCO_CLIP_R50.yaml --eval-only --num-gpus $GPUS$
```
- Open-vocabulary COCO
    |                       |  Novel | Base | All | Model |
    |-----------------------|--------|------|-----|-------|
    |MEDet                  | 32.6   | 53.5 | 48.0|[Model](https://drive.google.com/file/d/1bp6ripmONN0N5Lfmv9PkrbGdgO3gxl9r/view?usp=sharing)       |

### Acknowledgement
Our code is developed from [Detic](https://github.com/facebookresearch/Detic)
