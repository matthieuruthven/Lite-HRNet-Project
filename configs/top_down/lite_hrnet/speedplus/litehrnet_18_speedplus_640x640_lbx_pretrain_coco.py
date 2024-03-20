log_level = 'INFO'
load_from = 'litehrnet_18_coco_384x288.pth'
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=150)
evaluation = dict(interval=1, metric='PCKh', key_indicator='PCKh')

optimizer = dict(
    type='Adam',
    lr=2e-3,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[45, 60])
total_epochs = 100
log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=11,
    dataset_joints=11,
    dataset_channel=list(range(11)),
    inference_channel=list(range(11)))

# model settings
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='LiteHRNet',
        in_channels=3,
        extra=dict(
            stem=dict(  
                stem_channels=32,
                out_channels=32,
                expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                )),
            with_head=True,
            )),
    keypoint_head=dict(
        type='TopDownSimpleHead',
        in_channels=40,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process=True,
        shift_heatmap=True,
        unbiased_decoding=False,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[640, 640],
    heatmap_size=[160, 160],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    use_gt_bbox=True,
    bbox_file=None,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=['image_file']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['image_file', 'center', 'scale']),
]

data_root = '../speedplus_640'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='TopDownSpeedPlusDatasetPreProc',
        ann_file=f'{data_root}/lightbox/train_sda.json',
        img_prefix=f'{data_root}/lightbox/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='TopDownSpeedPlusDatasetPreProc',
        ann_file=f'{data_root}/lightbox/validation.json',
        img_prefix=f'{data_root}/lightbox/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='TopDownSpeedPlusDatasetPreProc',
        ann_file=f'{data_root}/lightbox/test.json',
        img_prefix=f'{data_root}/lightbox/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline)
)
