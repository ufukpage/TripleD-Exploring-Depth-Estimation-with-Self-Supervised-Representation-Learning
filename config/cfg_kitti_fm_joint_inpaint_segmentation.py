LAYERS = 50
IMGS_PER_GPU = 4
HEIGHT = 192#320
WIDTH = 640#1024

data = dict(
    name = 'cityscapes', #'cityscapes', # kitti
    split = 'train',
    height = HEIGHT,
    width = WIDTH,
    in_path = 'F:\\cityscapes' #'F:\\cityscapes', 'D:\\kitti_semantics'
)

model = dict(
    name='FixSegmentationDepth',
    use_loss_weights=True,
    num_layers = LAYERS,
    imgs_per_gpu = IMGS_PER_GPU,
    height = HEIGHT,
    width = WIDTH,
    fix_encoder=False,
    scales = [0, 1, 2, 3],
    pretrained_path='F:\\pretrained\\resnet{}.pth'.format(LAYERS),# pretrained weights for resnet
    extractor_pretrained_path = None, #'E:\\pretrained\\autoencoder.pth',# pretrained weights for autoencoder
)
validate = True
resume_from = None
finetune = 'D:\\epoch_40_inpaint.pth'
total_epochs = 30
imgs_per_gpu = IMGS_PER_GPU
learning_rate = 1e-4
workers_per_gpu = 1


optimizer = dict(type='Adam', lr=learning_rate, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    #warmup='linear',
    #warmup_iters=500,
    #warmup_ratio=1.0 / 3,
    step=[20],
    gamma=0.5,
)

num_classes = 20
checkpoint_config = dict(interval=5)
log_config = dict(interval=5,
                  hooks=[dict(type='TextLoggerHook'),])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# workflow = [('train', 1)]

workflow = [('train', 1)]