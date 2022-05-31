# openseg
Zero shot panoptic segmentation


For conditional prompt segmentation model training:
command - 
```bash
  python train_net.py --resume --config-file configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k.yaml --num-gpus 8 OUTPUT_DIR ./output/release/configs_coco-stuff-164k-156_zero_shot_maskformer_R101c_bs32_60k WANDB.NAME configs_coco-stuff-164k-156_zero_shot_maskformer_R101c_bs32_60k MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${PROMPT_MODEL} MODEL.CLIP_ADAPTER.PROMPT_SHAPE "(32, 0)"
  ```
