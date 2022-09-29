# faster-rcnn
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11001 tools/train.py \
configs/faster_rcnn/scale_backbone_lr/faster_rcnn_r101_fpn_1x_coco_r101_4421_0x5_t0x1const_lrmult0x5.py \
--work-dir work_dirs/faster_rcnn_r101_fpn_1x_coco_r101_4421_0x5_t0x1const_lrmult0x5 \
--launcher pytorch;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11002 tools/train.py \
configs/faster_rcnn/scale_backbone_lr/faster_rcnn_r101_fpn_1x_coco_r101_4471_0x5_t0x1const_lrmult0x5.py \
--work-dir work_dirs/faster_rcnn_r101_fpn_1x_coco_r101_4471_0x5_t0x1const_lrmult0x5 \
--launcher pytorch;

# retinanet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11003 tools/train.py \
configs/retinanet/scale_backbone_lr/retinanet_r101_fpn_1x_coco_r101_4421_0x5_t0x1const_lrmult0x2.py \
--work-dir work_dirs/retinanet_r101_fpn_1x_coco_r101_4421_0x5_t0x1const_lrmult0x2 \
--launcher pytorch;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11004 tools/train.py \
configs/retinanet/scale_backbone_lr/retinanet_r101_fpn_1x_coco_r101_4471_0x5_t0x1const_lrmult0x2.py \
--work-dir work_dirs/retinanet_r101_fpn_1x_coco_r101_4471_0x5_t0x1const_lrmult0x2 \
--launcher pytorch;

# mask-rcnn
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11005 tools/train.py \
configs/mask_rcnn/scale_backbone_lr/mask_rcnn_r101_fpn_1x_coco_r101_4421_0x4_t0x1const_lrmult0x5.py \
--work-dir work_dirs/mask_rcnn_r101_fpn_1x_coco_r101_4421_0x4_t0x1const_lrmult0x5 \
--launcher pytorch;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11005 tools/train.py \
configs/mask_rcnn/scale_backbone_lr/mask_rcnn_r101_fpn_1x_coco_r101_4471_0x4_t0x1const_lrmult0x5.py \
--work-dir work_dirs/mask_rcnn_r101_fpn_1x_coco_r101_4471_0x4_t0x1const_lrmult0x5 \
--launcher pytorch;