# Install

```
git clone https://github.com/DeepVoodooFX/GPEN.git
cd GPEN
virtualenv -p /usr/bin/python3.6 venv
. venv/bin/activate

pip install -r requirements.txt

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

cd weights
wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth
wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-512.pth
wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-1024-Color.pth

# Need to turn off cudnn for avoiding error
```


# Inference

```
# Run inference with DFL extracted/merged images and save the DFL meta data in the results
python face_enhancement.py \
--input_dir /ParkCounty/home/DFDNet_data/frank \
--output_dir /ParkCounty/home/DFDNet_data/frank_GPEN_512 \
--gpu_id 0


# Run inference with non-DFL images
python face_enhancement.py \
--input_dir /ParkCounty/home/DFDNet_data/frank \
--output_dir /ParkCounty/home/DFDNet_data/frank_GPEN_512 \
--data_type raw \
--gpu_id 0
```
