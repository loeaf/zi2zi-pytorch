 # 기존 패키지 제거
 conda create --name zi2zi-torch python=3.7
 conda activate zi2zi-torch
 pip uninstall torch torchvision torchaudio pillow numpy scipy imageio -y
 pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
 pip install pillow==7.1.2 numpy==1.18.1 scipy==1.4.1 imageio==2.8.0

# 단일 이미지 생성
python font2img.py --src_font=korean.ttf \
                   --dst_font=english.ttf \
                   --charset=custom_chars.txt \  # 공통 지원 문자만 포함
                   --sample_count=1000 \
                   --sample_dir=samples \
                   --label=0 \
                   --filter=1                    # 필터링 활성화


# create-dset.py
python create-dset.py --dir="/Users/doheyonkim/data/fontbox/dataset1" \
                      --save_dir="/Users/doheyonkim/data/fontbox/dataset1" \
                      --split_ratio=0.1

python create-dset-multi.py --dir="/Users/doheyonkim/data/fontbox/dataset1" \
                      --save_dir="/Users/doheyonkim/data/fontbox/dataset1" \
                      --split_ratio=0.1



python package.py \
    --dir="/Volumes/Extreme SSD/dataset" \
    --save_dir="/Volumes/Extreme SSD/dataset" \
    --split_ratio=0.9

python package-multi.py \
    --dir="/Users/doheyonkim/data/fontbox/dataset1" \
    --save_dir="/Users/doheyonkim/data/fontbox/dataset1" \
    --split_ratio=0.1



nohup python train.py \
  --experiment_dir=/data/dataset \
  --batch_size=200 \
  --lr=0.001 \
  --epoch=90000 \
  --sample_steps=100 \
  --schedule=20 \
  --L1_penalty=100 \
  --Lconst_penalty=15 \
  --embedding_num=4 \
  --gpu_ids 0 1 &


do \
  echo "Processing label $i..."; \
  python infer.py \
  --experiment_dir /home/user/data/dataset \
  --batch_size 32 \
  --resume 92000 \
  --from_txt \
  --src_font /home/user/data/dataset/d461cae31481b059a9e85df14fffe0d5 \
  --src_txt 가나다라마바사 \
  --label $i \
  --input_nc 3 \
  --embedding_num 704; \
  done
