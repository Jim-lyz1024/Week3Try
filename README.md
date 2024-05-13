# Vision-Language Generalizable Intelligence Library

## Execute code instructions
1. Install the required packages by running the following command:
```bash
pip install -r requirements.txt
```
2. Run the following command to execute the code:

PACS dataset:
```bash
python train.py --gpu 0 --seed 134 --output-dir output/CLIPAdapters-ViTB32-PACS-art_painting --dataset PACS --source-domains cartoon photo sketch --target-domains art_painting            --model CLIPAdapters --model-config-file config/clipadapters.yaml

python train.py --gpu 0 --seed 134 --output-dir output/CLIPAdapters-ViTB32-PACS-art_painting --dataset PACS --source-domains art_painting photo sketch --target-domains cartoon            --model CLIPAdapters --model-config-file config/clipadapters.yaml

python train.py --gpu 0 --seed 134 --output-dir output/CLIPAdapters-ViTB32-PACS-art_painting --dataset PACS --source-domains cartoon art_painting sketch --target-domains photo            --model CLIPAdapters --model-config-file config/clipadapters.yaml

python train.py --gpu 0 --seed 134 --output-dir output/CLIPAdapters-ViTB32-PACS-art_painting --dataset PACS --source-domains cartoon photo art_painting --target-domains sketch            --model CLIPAdapters --model-config-file config/clipadapters.yaml
```

OfficeHome:
```bash
python train.py --gpu 0 --seed 134 --output-dir output/CLIPAdapters-ViTB32-OfficeHome-art --dataset OfficeHome --source-domains clipart product real_world --target-domains art            --model CLIPAdapters --model-config-file config/clipadapters.yaml

python train.py --gpu 0 --seed 134 --output-dir output/CLIPAdapters-ViTB32-OfficeHome-art --dataset OfficeHome --source-domains art product real_world --target-domains clipart            --model CLIPAdapters --model-config-file config/clipadapters.yaml

python train.py --gpu 0 --seed 134 --output-dir output/CLIPAdapters-ViTB32-OfficeHome-art --dataset OfficeHome --source-domains art clipart real_world --target-domains product            --model CLIPAdapters --model-config-file config/clipadapters.yaml

python train.py --gpu 0 --seed 134 --output-dir output/CLIPAdapters-ViTB32-OfficeHome-art --dataset OfficeHome --source-domains art clipart product --target-domains real_world            --model CLIPAdapters --model-config-file config/clipadapters.yaml
```

VLCS dataset:
```bash
python train.py --gpu 0 --seed 134 --output-dir output/CLIPAdapters-ViTB32-VLCS-labelme --dataset VLCS --source-domains labelme pascal sun --target-domains caltech            --model CLIPAdapters --model-config-file config/clipadapters.yaml

python train.py --gpu 0 --seed 134 --output-dir output/CLIPAdapters-ViTB32-VLCS-labelme --dataset VLCS --source-domains caltech pascal sun --target-domains labelme            --model CLIPAdapters --model-config-file config/clipadapters.yaml

python train.py --gpu 0 --seed 134 --output-dir output/CLIPAdapters-ViTB32-VLCS-labelme --dataset VLCS --source-domains caltech labelme sun --target-domains pascal            --model CLIPAdapters --model-config-file config/clipadapters.yaml

python train.py --gpu 0 --seed 134 --output-dir output/CLIPAdapters-ViTB32-VLCS-labelme --dataset VLCS --source-domains caltech labelme pascal --target-domains sun            --model CLIPAdapters --model-config-file config/clipadapters.yaml
```

Digits dataset:
```bash
python train.py --gpu 0 --seed 134 --output-dir output/CLIPAdapters-ViTB32-Digits-mnist --dataset Digits --source-domains mnist_m svhn syn --target-domains mnist            --model CLIPAdapters --model-config-file config/clipadapters.yaml
```

3. Five seeds: **134, 232, 607, 779, 995** are used for the experiments. The results are saved in the output directory.

4. Fill in the table with the results obtained from the output directory:
   https://docs.google.com/spreadsheets/d/1OZXjCjE8OKRF_Ce-B-cyx3apKZreNTWA_9ncuXAn7Rs/edit#gid=0

