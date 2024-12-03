 nohup python train.py \
  --experiment_dir=/data/dataset \
  --batch_size=220 \
  --lr=0.001 \
  --epoch=1000 \
  --sample_steps=100 \
  --schedule=20 \
  --L1_penalty=100 \
  --Lconst_penalty=15 \
  --embedding_num=704 \
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
