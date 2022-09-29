CUDA_VISIBLE_DEVICES=9 python -u tools/test_get_info.py \
configs/faster_rcnn/scale_backbone_lr/faster_rcnn_r101_fpn_1x_coco_r101_4421_0x5_t0x1const_lrmult0x5.py \
work_dirs/faster_rcnn_r101_fpn_1x_coco_r101_4421_0x5_t0x1const_lrmult0x5/epoch_12.pth \
--work-dir work_dirs/faster_rcnn_r101_fpn_1x_coco_r101_4421_0x5_t0x1const_lrmult0x5_eval \
--eval bbox

CUDA_VISIBLE_DEVICES=9 python -u tools/test_get_info.py \
configs/faster_rcnn/scale_backbone_lr/faster_rcnn_r101_fpn_1x_coco_r101_4471_0x5_t0x1const_lrmult0x5.py \
work_dirs/faster_rcnn_r101_fpn_1x_coco_r101_4471_0x5_t0x1const_lrmult0x5/epoch_12.pth \
--work-dir work_dirs/faster_rcnn_r101_fpn_1x_coco_r101_4471_0x5_t0x1const_lrmult0x5_eval \
--eval bbox

CUDA_VISIBLE_DEVICES=9 python -u tools/test_get_info.py \
configs/retinanet/scale_backbone_lr/retinanet_r101_fpn_1x_coco_r101_4421_0x5_t0x1const_lrmult0x2.py \
work_dirs/retinanet_r101_fpn_1x_coco_r101_4421_0x5_t0x1const_lrmult0x2/epoch_12.pth \
--work-dir work_dirs/retinanet_r101_fpn_1x_coco_r101_4421_0x5_t0x1const_lrmult0x2_eval \
--eval bbox

CUDA_VISIBLE_DEVICES=9 python -u tools/test_get_info.py \
configs/retinanet/scale_backbone_lr/retinanet_r101_fpn_1x_coco_r101_4471_0x5_t0x1const_lrmult0x2.py \
work_dirs/retinanet_r101_fpn_1x_coco_r101_4471_0x5_t0x1const_lrmult0x2/epoch_12.pth \
--work-dir work_dirs/retinanet_r101_fpn_1x_coco_r101_4471_0x5_t0x1const_lrmult0x2_eval \
--eval bbox

CUDA_VISIBLE_DEVICES=9 python -u tools/test_get_info.py \
configs/mask_rcnn/scale_backbone_lr/mask_rcnn_r101_fpn_1x_coco_r101_4421_0x4_t0x1const_lrmult0x5.py \
work_dirs/mask_rcnn_r101_fpn_1x_coco_r101_4421_0x4_t0x1const_lrmult0x5/epoch_12.pth \
--work-dir work_dirs/mask_rcnn_r101_fpn_1x_coco_r101_4421_0x4_t0x1const_lrmult0x5_eval \
--eval segm

CUDA_VISIBLE_DEVICES=9 python -u tools/test_get_info.py \
configs/mask_rcnn/scale_backbone_lr/mask_rcnn_r101_fpn_1x_coco_r101_4471_0x4_t0x1const_lrmult0x5.py \
work_dirs/mask_rcnn_r101_fpn_1x_coco_r101_4471_0x4_t0x1const_lrmult0x5/epoch_12.pth \
--work-dir work_dirs/mask_rcnn_r101_fpn_1x_coco_r101_4471_0x4_t0x1const_lrmult0x5_eval \
--eval segm