default_scope = "mmpose"

custom_imports = dict(
    imports=["mmpose.engine.optim_wrappers.layer_decay_optim_wrapper"],
    allow_failed_imports=False,
)

train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=5)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=32,
        layer_decay_rate=0.85,
        custom_keys={
            "bias": dict(decay_multi=0.0),
            "pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        },
    ),
    constructor="LayerDecayOptimWrapperConstructor",
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(type="LinearLR", begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(
        type="MultiStepLR",
        begin=0,
        end=120,
        milestones=[90, 110],
        gamma=0.1,
        by_epoch=True,
    ),
]

auto_scale_lr = dict(base_batch_size=128)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=5,
        save_best="coco/AP",
        rule="greater",
        max_keep_ckpts=2,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="PoseLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True, num_digits=6)
log_level = "INFO"
load_from = None
resume = False

codec = dict(type="UDPHeatmap", input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

NUM_KEYPOINTS = 18

model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="mmpretrain.VisionTransformer",
        arch="huge",
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.55,
        with_cls_token=False,
        out_type="featmap",
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type="Pretrained",
            checkpoint=(
                "https://download.openmmlab.com/mmpose/"
                "v1/pretrained_models/mae_pretrain_vit_huge_20230913.pth"
            ),
        ),
    ),
    head=dict(
        type="HeatmapHead",
        in_channels=1280,
        out_channels=NUM_KEYPOINTS,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode="heatmap",
        shift_heatmap=False,
    ),
)

dataset_type = "CocoDataset"
data_mode = "topdown"
backend_args = dict(backend="local")

import os as _os

# MMEngine executes config files without defining __file__; fencing_inference sets
# FENCING18_METAINFO before loading this config.
metainfo_file = _os.environ.get(
    "FENCING18_METAINFO",
    "/app/mmpose_configs/fencing18_metainfo.py",
)
data_root = _os.environ.get("FENCING18_DATA_ROOT", "./vitpose_fullframes_output")

train_pipeline = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(type="RandomBBoxTransform"),
    dict(type="TopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

val_pipeline = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/train_bbox_expand.json",
        data_prefix=dict(img=""),
        metainfo=dict(from_file=metainfo_file),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/val_bbox_expand.json",
        data_prefix=dict(img=""),
        metainfo=dict(from_file=metainfo_file),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=r"c:\Users\jorda\Desktop\fencing\vitpose_fullframes_output\annotations\val_bbox_expand.json",
)
test_evaluator = val_evaluator

# Training / checkpoint folder aligned with app default (see mmpose_paths.fencing18_vitpose_work_dir).
work_dir = "./work_dirs/vitpose_huge_fencing18_256x192_stable17"
