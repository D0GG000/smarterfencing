# Minimal MMPose config to init Topdown ViTPose-H (17 kpts, 256x192) + COCO weights.
# Used by train_bellguard_fusion.py / infer_fencing_fusion18.py (frozen backbone + head).

default_scope = "mmpose"

codec = dict(type="UDPHeatmap", input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# Required by infer_fencing_fusion18.py (PoseLocalVisualizer / radius / line_width)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="PoseLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)

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
        out_channels=17,
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

# inference_topdown() reads this pipeline from model.cfg
test_dataloader = dict(
    dataset=dict(
        type="CocoDataset",
        pipeline=[
            dict(type="LoadImage"),
            dict(type="GetBBoxCenterScale"),
            dict(type="TopdownAffine", input_size=(192, 256), use_udp=True),
            dict(type="PackPoseInputs"),
        ],
    ),
)
