# HuBMAP: Hacking the Kidney Identify glomeruli in human kidney tissue images
* Website: https://www.kaggle.com/c/hubmap-kidney-segmentation
* Download the data into the directory `/data/hubmap-kidney-segmentation`

## Dependencies
* Install Anaconda and use the provided environment:
  ```
  conda env create -f environment.yml
  conda activate hacking_kidney
  ```

## Models
Trained model:
* `hacking_kidney_16934_best_metric.model-384e1332.pth`
  * single fold/model kaggle LB: 0.873
  * input patch 1024x1024
  * semi supervised [UNet](https://arxiv.org/abs/1505.04597) with [SCSE](https://arxiv.org/abs/1803.02579) using Resnet34 as backbone: [nn/unet.py](nn/unet.py)
  * streamlit demo: `streamlit run demo.py -- --image-size=1024 --mode=valid --model hacking_kidney_16934_best_metric.model-384e1332.pth`
    * it will visualize the validation dataset overlaying predictions and masks.
  * example training parameters for supervised learning on 8 GPUs:
  ```
  python -m torch.distributed.launch --nproc_per_node 8 train.py --data-root /data/hubmap-kidney-segmentation --jobs=40 \
      --frozen-batchnorm=false --max-epochs=100 --pretrained=imagenet --batch-size-per-gpu=true --batch-size=20 --image-size=1024 --resize=1024 \
      --optim=adamw --learning-rate=3e-4 --weight-decay=0.0 --data-fold=0 --loss-ce=1 --loss-dice=0 --loss-lovasz=0 --apex-opt-level=O2 --sync-bn=True \
      --arch=unet_scse --backbone=Resnet34 \
      --data-aug-image-compression-p=0.3 \
      --data-aug-gauss-noise-p=0.3 \
      --data-aug-gaussian-blur-p=0.3 \
      --data-aug-rgb-aug-p=0.3 \
      --data-aug-color-jitter-p=0.4 \
      --data-aug-rotate-p=0.4 \
      --data-aug-random-scale-p=0.3 \
      --data-aug-clahe-p=0.2 \
      --data-aug-distort-p=0.7
  ```
  Demo screenshot:
  ![](demo_screenshot.png)
  
## Running the demo
1. Download the model from [NextCloud](https://nx9836.your-storageshare.de/s/HSq8StKLB6WYncy) (default directory: project folder). **Do not distribute**.
2. Download the kaggle data (default directory: `./data/hubmap-kidney-segmentation`) and ensure that the directory contains the following files:
    ```
    tree -L 1 /data/hubmap-kidney-segmentation
    /data/hubmap-kidney-segmentation
    ├── HuBMAP-20-dataset_information.csv
    ├── sample_submission.csv
    ├── test
    ├── train
    └── train.csv
    ```
2. Build docker image (we are using `nvcr.io/nvidia/pytorch:20.07-py3` which contains PyTorch 1.6 and Anaconda packages):
  ```
  ./build_container.sh
  ```

3. Run the docker container. Optional arguments may be provided if data or model are located in non-default directories.
  ```
  ./start_container.sh [-d data_path] [-m model_path] [-t test_suite_path]
  ```

4. Access the UI on http://localhost:8501/
