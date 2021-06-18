# Install

```
git clone https://github.com/chuanli11/GPEN.git
cd GPEN
virtualenv -p /usr/bin/python3.6 venv
. venv/bin/activate

pip install -r requirements.txt
pip uninstall torch
pip install torch==1.7.0 torchvision==0.8.0

cd weights
wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth
wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-512.pth
wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-1024-Color.pth

# Need to turn off cudnn for avoiding error
```


# Inference

```
python face_enhancement.py \
--input_dir /ParkCounty/home/DFDNet_data/frank \
--output_dir /ParkCounty/home/DFDNet_data/frank_GPEN_512 \
--gpu_id 0
```