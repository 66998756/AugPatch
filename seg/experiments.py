# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import itertools
import os

from mmcv import Config

# flake8: noqa


def get_model_base(architecture, backbone):
    architecture = architecture.replace('sfa_', '')
    for j in range(1, 100):
        hrda_name = [e for e in architecture.split('_') if f'hrda{j}' in e]
        for n in hrda_name:
            architecture = architecture.replace(f'{n}_', '')
    architecture = architecture.replace('_nodbn', '')
    if 'segformer' in architecture:
        return {
            'mitb5': f'_base_/models/{architecture}_b5.py',
            # It's intended that <=b4 refers to b5 config
            'mitb4': f'_base_/models/{architecture}_b5.py',
            'mitb3': f'_base_/models/{architecture}_b5.py',
            'r101v1c': f'_base_/models/{architecture}_r101.py',
        }[backbone]
    if 'daformer_' in architecture and 'mitb5' in backbone:
        return f'_base_/models/{architecture}_mitb5.py'
    if 'upernet' in architecture and 'mit' in backbone:
        return f'_base_/models/{architecture}_mit.py'
    assert 'mit' not in backbone or '-del' in backbone
    return {
        'dlv2': '_base_/models/deeplabv2_r50-d8.py',
        'dlv2red': '_base_/models/deeplabv2red_r50-d8.py',
        'dlv3p': '_base_/models/deeplabv3plus_r50-d8.py',
        'da': '_base_/models/danet_r50-d8.py',
        'isa': '_base_/models/isanet_r50-d8.py',
        'uper': '_base_/models/upernet_r50.py',
    }[architecture]


def get_pretraining_file(backbone):
    if 'mitb5' in backbone:
        return 'pretrained/mit_b5.pth'
    if 'mitb4' in backbone:
        return 'pretrained/mit_b4.pth'
    if 'mitb3' in backbone:
        return 'pretrained/mit_b3.pth'
    if 'r101v1c' in backbone:
        return 'open-mmlab://resnet101_v1c'
    return {
        'r50v1c': 'open-mmlab://resnet50_v1c',
        'x50-32': 'open-mmlab://resnext50_32x4d',
        'x101-32': 'open-mmlab://resnext101_32x4d',
        's50': 'open-mmlab://resnest50',
        's101': 'open-mmlab://resnest101',
        's200': 'open-mmlab://resnest200',
    }[backbone]


def get_backbone_cfg(backbone):
    for i in [1, 2, 3, 4, 5]:
        if backbone == f'mitb{i}':
            return dict(type=f'mit_b{i}')
        if backbone == f'mitb{i}-del':
            return dict(_delete_=True, type=f'mit_b{i}')
    return {
        'r50v1c': {
            'depth': 50
        },
        'r101v1c': {
            'depth': 101
        },
        'x50-32': {
            'type': 'ResNeXt',
            'depth': 50,
            'groups': 32,
            'base_width': 4,
        },
        'x101-32': {
            'type': 'ResNeXt',
            'depth': 101,
            'groups': 32,
            'base_width': 4,
        },
        's50': {
            'type': 'ResNeSt',
            'depth': 50,
            'stem_channels': 64,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's101': {
            'type': 'ResNeSt',
            'depth': 101,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's200': {
            'type': 'ResNeSt',
            'depth': 200,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True,
        },
    }[backbone]


def update_decoder_in_channels(cfg, architecture, backbone):
    cfg.setdefault('model', {}).setdefault('decode_head', {})
    if 'dlv3p' in architecture and 'mit' in backbone:
        cfg['model']['decode_head']['c1_in_channels'] = 64
    if 'sfa' in architecture:
        cfg['model']['decode_head']['in_channels'] = 512
    return cfg


def setup_rcs(cfg, temperature, min_crop_ratio):
    cfg.setdefault('data', {}).setdefault('train', {})
    cfg['data']['train']['rare_class_sampling'] = dict(
        min_pixels=3000, class_temp=temperature, min_crop_ratio=min_crop_ratio)
    return cfg


def generate_experiment_cfgs(id):

    def config_from_vars():
        cfg = {
            '_base_': ['_base_/default_runtime.py'],
            'gpu_model': gpu_model,
            'n_gpus': n_gpus
        }
        if seed is not None:
            cfg['seed'] = seed

        # Setup model config
        architecture_mod = architecture
        sync_crop_size_mod = sync_crop_size
        inference_mod = inference
        model_base = get_model_base(architecture_mod, backbone)
        model_base_cfg = Config.fromfile(os.path.join('configs', model_base))
        cfg['_base_'].append(model_base)
        cfg['model'] = {
            'pretrained': get_pretraining_file(backbone),
            'backbone': get_backbone_cfg(backbone),
        }
        if 'sfa_' in architecture_mod:
            cfg['model']['neck'] = dict(type='SegFormerAdapter')
        if '_nodbn' in architecture_mod:
            cfg['model'].setdefault('decode_head', {})
            cfg['model']['decode_head']['norm_cfg'] = None
        cfg = update_decoder_in_channels(cfg, architecture_mod, backbone)

        hrda_ablation_opts = None
        outer_crop_size = sync_crop_size_mod \
            if sync_crop_size_mod is not None \
            else (int(crop.split('x')[0]), int(crop.split('x')[1]))
        if 'hrda1' in architecture_mod:
            o = [e for e in architecture_mod.split('_') if 'hrda' in e][0]
            hr_crop_size = (int((o.split('-')[1])), int((o.split('-')[1])))
            hr_loss_w = float(o.split('-')[2])
            hrda_ablation_opts = o.split('-')[3:]
            cfg['model']['type'] = 'HRDAEncoderDecoder'
            cfg['model']['scales'] = [1, 0.5]
            cfg['model'].setdefault('decode_head', {})
            cfg['model']['decode_head']['single_scale_head'] = model_base_cfg[
                'model']['decode_head']['type']
            cfg['model']['decode_head']['type'] = 'HRDAHead'
            cfg['model']['hr_crop_size'] = hr_crop_size
            cfg['model']['feature_scale'] = 0.5
            cfg['model']['crop_coord_divisible'] = 8
            cfg['model']['hr_slide_inference'] = True
            cfg['model']['decode_head']['attention_classwise'] = True
            cfg['model']['decode_head']['hr_loss_weight'] = hr_loss_w
            if outer_crop_size == hr_crop_size:
                # If the hr crop is smaller than the lr crop (hr_crop_size <
                # outer_crop_size), there is direct supervision for the lr
                # prediction as it is not fused in the region without hr
                # prediction. Therefore, there is no need for a separate
                # lr_loss.
                cfg['model']['decode_head']['lr_loss_weight'] = hr_loss_w
                # If the hr crop covers the full lr crop region, calculating
                # the FD loss on both scales stabilizes the training for
                # difficult classes.
                cfg['model']['feature_scale'] = 'all' if '_fd' in uda else 0.5

        # HRDA Ablations
        if hrda_ablation_opts is not None:
            for o in hrda_ablation_opts:
                if o == 'fixedatt':
                    # Average the predictions from both scales instead of
                    # learning a scale attention.
                    cfg['model']['decode_head']['fixed_attention'] = 0.5
                elif o == 'nooverlap':
                    # Don't use overlapping slide inference for the hr
                    # prediction.
                    cfg['model']['hr_slide_overlapping'] = False
                elif o == 'singleatt':
                    # Use the same scale attention for all class channels.
                    cfg['model']['decode_head']['attention_classwise'] = False
                elif o == 'blurhr':
                    # Use an upsampled lr crop (blurred) for the hr crop
                    cfg['model']['blur_hr_crop'] = True
                elif o == 'samescale':
                    # Use the same scale/resolution for both crops.
                    cfg['model']['scales'] = [1, 1]
                    cfg['model']['feature_scale'] = 1
                elif o[:2] == 'sc':
                    cfg['model']['scales'] = [1, float(o[2:])]
                    if not isinstance(cfg['model']['feature_scale'], str):
                        cfg['model']['feature_scale'] = float(o[2:])
                else:
                    raise NotImplementedError(o)

        # Setup inference mode
        if inference_mod == 'whole' or crop == '2048x1024':
            assert model_base_cfg['model']['test_cfg']['mode'] == 'whole'
        elif inference_mod == 'slide':
            cfg['model'].setdefault('test_cfg', {})
            cfg['model']['test_cfg']['mode'] = 'slide'
            cfg['model']['test_cfg']['batched_slide'] = True
            crsize = sync_crop_size_mod if sync_crop_size_mod is not None \
                else [int(e) for e in crop.split('x')]
            cfg['model']['test_cfg']['stride'] = [e // 2 for e in crsize]
            cfg['model']['test_cfg']['crop_size'] = crsize
            architecture_mod += '_sl'
        else:
            raise NotImplementedError(inference_mod)

        # Setup UDA config
        if uda == 'target-only':
            cfg['_base_'].append(f'_base_/datasets/{target}_{crop}.py')
        elif uda == 'source-only':
            cfg['_base_'].append(
                f'_base_/datasets/{source}_to_{target}_{crop}.py')
        else:
            cfg['_base_'].append(
                f'_base_/datasets/uda_{source}_to_{target}_{crop}.py')
            cfg['_base_'].append(f'_base_/uda/{uda}.py')
        cfg['data'] = dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=workers_per_gpu,
            train={})
        # DAFormer legacy cropping that only works properly if the training
        # crop has the height of the (resized) target image.
        if ('dacs' in uda or mask_mode is not None or aug_mode is not None) \
            and plcrop in [True, 'v1']:
            cfg.setdefault('uda', {})
            cfg['uda']['pseudo_weight_ignore_top'] = 15
            cfg['uda']['pseudo_weight_ignore_bottom'] = 120
        # Generate mask of the pseudo-label margins in the data loader before
        # the image itself is cropped to ensure that the pseudo-label margins
        # are only masked out if the training crop is at the periphery of the
        # image.
        if ('dacs' in uda or mask_mode is not None or aug_mode is not None) \
            and plcrop == 'v2':
            cfg['data']['train'].setdefault('target', {})
            cfg['data']['train']['target']['crop_pseudo_margins'] = \
                [30, 240, 30, 30]
        if 'dacs' in uda and rcs_T is not None:
            cfg = setup_rcs(cfg, rcs_T, rcs_min_crop)
        if 'dacs' in uda and sync_crop_size_mod is not None:
            cfg.setdefault('data', {}).setdefault('train', {})
            cfg['data']['train']['sync_crop_size'] = sync_crop_size_mod
        if mask_mode is not None:
            cfg.setdefault('uda', {})
            cfg['uda']['mask_mode'] = mask_mode
            cfg['uda']['mask_alpha'] = mask_alpha
            cfg['uda']['mask_pseudo_threshold'] = mask_pseudo_threshold
            cfg['uda']['mask_lambda'] = mask_lambda
            cfg['uda']['mask_generator'] = dict(
                type='block',
                mask_ratio=mask_ratio,
                mask_block_size=mask_block_size,
                _delete_=True)
        if aug_mode is not None:
            cfg.setdefault('uda', {})
            cfg['uda']['patch_augment'] = {
                'aug_mode': aug_mode,
                'aug_alpha': aug_alpha,
                'aug_pseudo_threshold': aug_pseudo_threshold,
                'aug_lambda': aug_lambda,
                'aug_generator': dict(
                    type=aug_type,
                    augment_setup=augment_setup,
                    num_diff_aug=num_diff_aug, 
                    aug_block_size=aug_block_size, 
                    _delete_=True),
                'geometric_perturb': geometric_perturb,
                'mixing_cfg': mixing_cfg,
                'cls_mask': cls_mask,
                'consis_mode': consis_mode
            }
        # Self-voting setup
        cfg['uda']['refine'] = None
        if enable_refine:
            refine_cfg = {
                'k': k,
                'refine_aug': {
                    'type': 'RandAugment',
                    'augment_setup': refine_aug,
                    'num_diff_aug': num_diff_ref_aug,
                    'aug_block_size': 0},
                'start_iters': start_iters,
                'max_bank_size': max_bank_size,
                'refine_conf': refine_conf}
            cfg['uda']['refine'] = refine_cfg
        
        # 其他 Setup
        cfg['uda']['loss_adjustment'] = loss_adjustment
        # cfg['uda']['debug_img_interval'] = iters // 40
        cfg['uda']['debug_img_interval'] = 1000

        # Setup optimizer and schedule
        if 'dacs' in uda or 'minent' in uda or 'advseg' in uda:
            cfg['optimizer_config'] = None  # Don't use outer optimizer

        cfg['_base_'].extend(
            [f'_base_/schedules/{opt}.py', f'_base_/schedules/{schedule}.py'])
        cfg['optimizer'] = {'lr': lr}
        cfg['optimizer'].setdefault('paramwise_cfg', {})
        cfg['optimizer']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optimizer']['paramwise_cfg']['custom_keys']
        if pmult:
            opt_param_cfg['head'] = dict(lr_mult=10.)
        if 'mit' in backbone:
            opt_param_cfg['pos_block'] = dict(decay_mult=0.)
            opt_param_cfg['norm'] = dict(decay_mult=0.)

        # Setup runner
        cfg['runner'] = dict(type='IterBasedRunner', max_iters=iters)
        cfg['checkpoint_config'] = dict(
            by_epoch=False, interval=iters, max_keep_ckpts=1)
        # cfg['evaluation'] = dict(interval=iters // 40, metric='mIoU')
        cfg['evaluation'] = dict(interval=1000, metric='mIoU')

        # Construct config name
        uda_mod = uda
        if 'dacs' in uda and rcs_T is not None:
            uda_mod += f'_rcs{rcs_T}'
            if rcs_min_crop != 0.5:
                uda_mod += f'-{rcs_min_crop}'
        if 'dacs' in uda and sync_crop_size_mod is not None:
            uda_mod += f'_sf{sync_crop_size_mod[0]}x{sync_crop_size_mod[1]}'
        if 'dacs' in uda or mask_mode is not None:
            if not plcrop:
                pass
            elif plcrop in [True, 'v1']:
                uda_mod += '_cpl'
            elif plcrop[0] == 'v':
                uda_mod += f'_cpl{plcrop[1:]}'
            else:
                raise NotImplementedError(plcrop)
        if mask_mode is not None:
            uda_mod += f'_m{mask_block_size}-' \
                       f'{mask_ratio}-'
            if mask_alpha != 'same':
                uda_mod += f'a{mask_alpha}-'
            if mask_pseudo_threshold != 'same':
                uda_mod += f'p{mask_pseudo_threshold}-'
            uda_mod += {
                'separate': 'sep',
                'separateaug': 'spa',
                'separatesrc': 'sps',
                'separatesrcaug': 'spsa',
                'separatetrg': 'spt',
                'separatetrgaug': 'spta',
            }[mask_mode]
            if mask_lambda != 1:
                uda_mod += f'-w{mask_lambda}'
        crop_name = f'_{crop}' if crop != '512x512' else ''
        cfg['name'] = f'{source}2{target}{crop_name}_{uda_mod}_' \
                      f'{architecture_mod}_{backbone}_{schedule}'
        if opt != 'adamw':
            cfg['name'] += f'_{opt}'
        if lr != 0.00006:
            cfg['name'] += f'_{lr}'
        if not pmult:
            cfg['name'] += f'_pm{pmult}'
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}{crop_name}'
        cfg['name_architecture'] = f'{architecture_mod}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture_mod
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' \
                          f'_{n_gpus}x{batch_size}_{iters // 1000}k'
        if seed is not None:
            cfg['name'] += f'_s{seed}'
        cfg['name'] = cfg['name'].replace('.', '').replace('True', 'T') \
            .replace('False', 'F').replace('None', 'N').replace('[', '')\
            .replace(']', '').replace(',', 'j').replace(' ', '') \
            .replace('cityscapes', 'cs') \
            .replace('synthia', 'syn') \
            .replace('darkzurich', 'dzur')
        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    cfgs = []
    n_gpus = 1
    batch_size = 2
    iters = 40000
    opt, lr, schedule, pmult = 'adamw', 0.00006, 'poly10warm', True
    crop = '512x512'
    gpu_model = 'NVIDIAGeForceRTX2080Ti'
    datasets = [
        ('gta', 'cityscapes'),
    ]
    architecture = None
    workers_per_gpu = 1
    rcs_T = None
    rcs_min_crop = 0.5
    plcrop = False
    inference = 'whole'
    sync_crop_size = None

    # mask init
    mask_mode = None
    mask_alpha = 'same'
    mask_pseudo_threshold = 'same'
    mask_lambda = 1
    mask_block_size = None
    mask_ratio = 0

    # AugPatch init
    aug_mode = None
    aug_alpha = 'same'
    aug_pseudo_threshold = 'same'
    aug_lambda = 1.0
    # aug_generator setup
    aug_type = 'RandAugment'
    augment_setup = {'n': 8, 'm': 20}
    num_diff_aug = None
    aug_block_size = None
    # apply class masking
    cls_mask = 'Random'
    # other setup
    geometric_perturb = None
    loss_adjustment = False

    # Self-voting setup
    enable_refine = False
    k = 4
    refine_aug = {'n': 2, 'm': 10}
    num_diff_ref_aug = 16
    # 從1500 iter 開始儲存，在2000 iter開始進行refine
    # 開始儲存最新feature的時間點是 
    # start_iters - (max_bank_size / batch size)
    start_iters = 2000
    max_bank_size = 1000
    # MIC like
    refine_conf = 0.968

    # patch mixing
    patch_mixing = False

    # -------------------------------------------------------------------------
    # MIC with HRDA for Different UDA Benchmarks (Table 2)
    # -------------------------------------------------------------------------
    # yapf: disable
    if id == 80:
        seeds = [0, 1, 2]
        architecture, backbone = 'hrda1-512-0.1_daformer_sepaspp', 'mitb5'
        uda, rcs_T = 'dacs_a999_fdthings', 0.01
        crop, rcs_min_crop = '1024x1024', 0.5 * (2 ** 2)
        inference = 'slide'
        mask_block_size, mask_ratio = 64, 0.7
        for source,          target,         mask_mode in [
            ('gtaHR',        'cityscapesHR', 'separatetrgaug'),
            ('synthiaHR',    'cityscapesHR', 'separatetrgaug'),
            ('cityscapesHR', 'acdcHR',       'separate'),
            ('cityscapesHR', 'darkzurichHR', 'separate'),
        ]:
            for seed in seeds:
                gpu_model = 'NVIDIATITANRTX'
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = 'v2' if 'cityscapes' in target else False
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # MIC with Further UDA Methods (Table 1)
    # -------------------------------------------------------------------------
    elif id == 81:
        seeds = [0, 1, 2]
        #        opt,     lr,      schedule,     pmult
        sgd   = ('sgd',   0.0025,  'poly10warm', False)
        adamw = ('adamw', 0.00006, 'poly10warm', True)
        #               uda,                  rcs_T, plcrop, opt_hp
        uda_advseg =   ('advseg',             None,  False,  *sgd)
        uda_minent =   ('minent',             None,  False,  *sgd)
        uda_dacs =     ('dacs',               None,  False,  *adamw)
        uda_daformer = ('dacs_a999_fdthings', 0.01,  True,   *adamw)
        uda_hrda =     ('dacs_a999_fdthings', 0.01,  'v2',   *adamw)
        mask_mode, mask_ratio = 'separatetrgaug', 0.7
        for architecture,                      backbone,  uda_hp in [
            # ('dlv2red',                        'r101v1c', uda_advseg),
            # ('dlv2red',                        'r101v1c', uda_minent),
            # ('dlv2red',                        'r101v1c', uda_dacs),
            # ('dlv2red',                        'r101v1c', uda_daformer),
            # ('hrda1-512-0.1_dlv2red',          'r101v1c', uda_hrda),
            ('daformer_sepaspp',               'mitb5',   uda_daformer),
            # ('hrda1-512-0.1_daformer_sepaspp', 'mibt5',   uda_hrda),  # already run in exp 80
        ]:
            if 'hrda' in architecture:
                source, target, crop = 'gtaHR', 'cityscapesHR', '1024x1024'
                rcs_min_crop = 0.5 * (2 ** 2)
                gpu_model = 'NVIDIATITANRTX'
                inference = 'slide'
                mask_block_size = 64
            else:
                source, target, crop = 'cityscapes', 'acdc', '512x512'
                rcs_min_crop = 0.5
                gpu_model = 'NVIDIAGeForceRTX2080Ti'
                inference = 'whole'
                # Use half the patch size when training with half resolution
                mask_block_size = 32
                # AugPatch init
                aug_mode = None
                aug_alpha = 'same'
                aug_pseudo_threshold = 'same'
                aug_lambda = 1.0
            for seed in seeds:
                uda, rcs_T, plcrop, opt, lr, schedule, pmult = uda_hp
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # MIC Domain Study (Table 6)
    # -------------------------------------------------------------------------
    elif id == 82:
        seeds = [0, 1, 2]
        architecture, backbone = 'hrda1-512-0.1_daformer_sepaspp', 'mitb5'
        uda, rcs_T = 'dacs_a999_fdthings', 0.01
        crop, rcs_min_crop = '1024x1024', 0.5 * (2 ** 2)
        inference = 'slide'
        mask_block_size, mask_ratio = 64, 0.7
        for source, target, mask_mode in [
            ('gtaHR', 'cityscapesHR',  'separatesrcaug'),
            # ('gtaHR', 'cityscapesHR',  'separatetrgaug'),  # already run in exp 80
            ('gtaHR', 'cityscapesHR',  'separateaug'),
            ('cityscapesHR', 'acdcHR', 'separatesrc'),
            ('cityscapesHR', 'acdcHR', 'separatetrg'),
            # ('cityscapesHR', 'acdcHR', 'separate'),  # already run in exp 80
        ]:
            for seed in seeds:
                gpu_model = 'NVIDIATITANRTX'
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = 'v2' if 'cityscapes' in target else False
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # MIC Ablation Study (Table 7)
    # -------------------------------------------------------------------------
    elif id == 83:
        seeds = [0, 1, 2]
        source, target = 'gta', 'cityscapes'
        architecture, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, True
        masking = [
            # mode,            b,  r,   a,      tau
            # ('separatetrgaug', 32, 0.7, 'same'),  # complete; already run in exp 81
            ('separatetrg',    32, 0.7, 'same', 'same'),  # w/o color augmentation
            ('separatetrgaug', 32, 0,   'same', 'same'),  # w/o masking
            ('separatetrgaug', 32, 0.7, 0,      'same'),  # w/o EMA teacher
            ('separatetrgaug', 32, 0.7, 'same', None),    # w/o pseudo label confidence weight
        ]
        for (mask_mode, mask_block_size, mask_ratio, mask_alpha, mask_pseudo_threshold), seed in \
                itertools.product(masking, seeds):
            if mask_alpha != 'same' or mask_pseudo_threshold != 'same':
                # Needs more gpu memory due to additional teacher
                gpu_model = 'NVIDIATITANRTX'
            else:
                gpu_model = 'NVIDIAGeForceRTX2080Ti'
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # MIC Block Size/Ratio (Table 8)
    # -------------------------------------------------------------------------
    elif id == 84:
        seeds = [0, 1, 2]
        source, target = 'gta', 'cityscapes'
        architecture, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, True
        # The patch sizes are divided by 2 here, as DAFormer uses half resolution
        block_sizes = [16, 32, 64, 128]
        ratios = [0.3, 0.5, 0.7, 0.9]
        mask_mode = 'separatetrgaug'
        for mask_block_size, mask_ratio, seed in \
                itertools.product(block_sizes, ratios, seeds):
            if mask_block_size == 32 and mask_ratio == 0.7:
                continue  # already run in exp 81
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # MIC for Supervised Training (Table 9 column mIoU_Superv.)
    # -------------------------------------------------------------------------
    elif id == 85:
        seeds = [0, 1, 2]
        architecture, backbone = 'daformer_sepaspp', 'mitb5'
        # Hack for supervised target training with MIC
        source, target, uda = 'cityscapes', 'cityscapes', 'dacs_srconly'
        mask_mode, mask_block_size, mask_ratio = 'separatesrc', 32, 0.7
        for seed in seeds:
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # AugPatch GTA2CS DAFormer
    # -------------------------------------------------------------------------
    elif id == 86:
        seeds = [0, 1, 2]
        source, target = 'gta', 'cityscapes'
        architecture, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, True

        # 由於 Rare Class 的表現不佳，懷疑是RCS參數不夠好，因此手動調整
        # 越小稀有類的 sample 機率越高，參考DAFormer Table S2.
        # rcs_T = 0.002

        architecture = 'daformer_sepaspp'
        rcs_min_crop = 0.5
        backbone = 'mitb5'

        gpu_model = 'NVIDIAGeForceRTX2080Ti'
        inference = 'whole'
         
        # gta to cityscapes
        source, target = 'gta', 'cityscapes'

        # MIC setup
        mask_block_size, mask_ratio = 32, 0.7
        mask_lambda = 1.0
        mask_mode = 'separatetrgaug'

        # AugPatch setup
        aug_mode = 'separatetrgaug'
        aug_lambda = 1.0
        num_diff_aug = 16
        augment_setup = {'n': 8, 'm': 30}
        geometric_perturb = True
        cls_mask = 'Random'

        # Self-voting setup
        enable_refine = False

        loss_adjustment = 1.5

        for seed in seeds:
            for block_size in [
                8,
                16,
                32,
                64,
                128
            ]:
                gpu_model = 'NVIDIARTX2080Ti'
                # balance lambda
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                aug_block_size = block_size
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # MIC with HRDA + aug patch
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 87:
        seeds = [0, 1, 2]
        source, target = 'cityscapes', 'acdc'
        architecture, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, False

        # 由於 Rare Class 的表現不佳，懷疑是RCS參數不夠好，因此手動調整
        # 越小稀有類的 sample 機率越高，參考DAFormer Table S2.
        # rcs_T = 0.002

        architecture = 'daformer_sepaspp'
        rcs_min_crop = 0.5
        backbone = 'mitb5'

        gpu_model = 'NVIDIAGeForceRTX2080Ti'
        inference = 'whole'

        # MIC setup
        mask_block_size, mask_ratio = 32, 0.7
        mask_lambda = 1.0
        mask_mode = 'separate'

        # AugPatch setup
        aug_mode = 'separateaug'
        aug_lambda = 1.0
        num_diff_aug = 8
        augment_setup = {'n': 8, 'm': 30}
        geometric_perturb = False
        cls_mask = 'Random'

        # Self-voting setup
        enable_refine = False

        loss_adjustment = 1.5

        for seed in seeds:
            for aug_times in [
                2,
                4,
                8,
                16,
                32
            ]:
                for block_size in [
                    8,
                    16,
                    32,
                    64,
                    128
                ]:
                    gpu_model = 'NVIDIARTX2080Ti'
                    # balance lambda
                    # plcrop is only necessary for Cityscapes as target domains
                    # ACDC and DarkZurich have no rectification artifacts.
                    aug_block_size, num_diff_aug = block_size, aug_times
                    cfg = config_from_vars()
                    cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # MIC with DAFormer AugPatch Implementation, cityscapes to acdc
    # -------------------------------------------------------------------------
    elif id == 88:
        seeds = [2]
        #        opt,     lr,      schedule,     pmult
        sgd   = ('sgd',   0.0025,  'poly10warm', False)
        adamw = ('adamw', 0.00006, 'poly10warm', True)
        #               uda,                  rcs_T, plcrop, opt_hp
        uda_daformer = ('dacs_a999_fdthings', 0.01,  False,   *adamw)

        uda, rcs_T, plcrop, opt, lr, schedule, pmult = uda_daformer

        architecture = 'daformer_sepaspp'
        rcs_min_crop = 0.5
        backbone = 'mitb5'

        gpu_model = 'NVIDIAGeForceRTX2080Ti'
        inference = 'whole'
         
        # cityscapes to acdc
        source, target = 'cityscapes', 'acdc'

        # MIC setup
        mask_block_size, mask_ratio = 32, 0.7
        mask_lambda = 1.0
        mask_mode = 'separatetrgaug'

        # AugPatch setup
        aug_mode = 'separatetrgaug'
        aug_lambda = 1.0
        aug_block_size = 32
        num_diff_aug = 8
        augment_setup={'n': 3, 'm': 10}

        loss_adjustment = 2

        # Self-voting setup
        enable_refine = False

        for loss_adjustment, geometric_perturb, cls_mask in [
            (True,           False,             'Random'),
            (True,           True,              'Random'),
            (True,           True,               False),

            # (False,          True,              False),
            # (True,           False,             False),

            # (False,          False,             False),
            # (False,          False,             'Random'),
            # (False,          True,              'Random'),
            # (True,           True,              'Random'),
        ]:
            for seed in seeds:
                gpu_model = 'NVIDIAA40'
                # balance lambda
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = 'v2' if 'cityscapes' in target else False
                plcrop = True
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # MIC with DAFormer AugPatch Implementation, cityscapes to darkzurich
    # -------------------------------------------------------------------------
    elif id == 89:
        seeds = [0, 1]
        #        opt,     lr,      schedule,     pmult
        sgd   = ('sgd',   0.0025,  'poly10warm', False)
        adamw = ('adamw', 0.00006, 'poly10warm', True)
        #               uda,                  rcs_T, plcrop, opt_hp
        uda_daformer = ('dacs_a999_fdthings', 0.01,  False,   *adamw)

        uda, rcs_T, plcrop, opt, lr, schedule, pmult = uda_daformer

        # 由於 Rare Class 的表現不佳，懷疑是RCS參數不夠好，因此手動調整
        # 越小稀有類的 sample 機率越高，參考DAFormer Table S2.
        # rcs_T = 0.002

        architecture = 'daformer_sepaspp'
        rcs_min_crop = 0.5
        backbone = 'mitb5'

        gpu_model = 'NVIDIAGeForceRTX2080Ti'
        inference = 'whole'
         
        # cityscapes to acdc (HRDA)
        source, target = 'cityscapes', 'acdc'
        # source, target = 'cityscapes', 'darkzurich'

        # MIC setup
        mask_block_size, mask_ratio = 32, 0.7
        mask_lambda = 1.0
        mask_mode = 'separate'
        # AugPatch setup

        aug_mode = 'separateaug'
        aug_lambda = 1.0
        aug_block_size = 32
        num_diff_aug = 8
        augment_setup = {'n': 8, 'm': 30}
        cls_mask = 'Random'

        # Self-voting setup
        enable_refine = False

        geometric_perturb = False
        loss_adjustment = 1.5

        for seed in seeds:
            for geometric_perturb in [
                False,
                True,
            ]:
                gpu_model = 'NVIDIARTX2080Ti'
                # balance lambda
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = True if 'cityscapes' in target else False
                # plcrop = True
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Real world UDA for AugPatch
    # -------------------------------------------------------------------------
    elif id == 90:
        seeds = [2]
        architecture, backbone = 'hrda1-512-0.1_daformer_sepaspp', 'mitb5'
        uda, rcs_T = 'dacs_a999_fdthings', 0.01
        crop, rcs_min_crop = '1024x1024', 0.5 * (2 ** 2)
        inference = 'slide'
        source, target = 'cityscapesHR', 'acdcHR'
        # source, target = 'cityscapesHR', 'darkzurichHR'
        # mask setup
        mask_mode = 'separatetrgaug'
        mask_block_size, mask_ratio = 64, 0.7
        mask_lambda = 0.5
        
        # AugPatch Detail setting
        aug_mode = 'separatetrgaug'
        aug_alpha = 'same'
        aug_pseudo_threshold = 'same'
        aug_lambda = 1.0
        # aug_generator setup
        aug_type = 'RandAugment'
        augment_setup = {'n': 8, 'm': 30}
        num_diff_aug = 16
        aug_block_size = 64
        # apply class masking
        cls_mask = 'Random'
        # geometric_perturb = False

        # # consistency setup
        # loss_adjustment = True

        for geometric_perturb, loss_adjustment in [
            (True, False),
            (True, True),
        ]:
            aug_lambda = 0.5 if loss_adjustment else 1.0
            for seed in seeds:
                gpu_model = 'NVIDIATITANRTX'
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = 'v2' if 'cityscapes' in target else False
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # MIC with HRDA AugPatch Implementation, cityscapes to ACDC (refer exp 82)
    # -------------------------------------------------------------------------
    elif id == 91:
        seeds = [2]
        architecture, backbone = 'hrda1-512-0.1_daformer_sepaspp', 'mitb5'
        uda, rcs_T = 'dacs_a999_fdthings', 0.01
        crop, rcs_min_crop = '1024x1024', 0.5 * (2 ** 2)
        inference = 'slide'

        # cityscapes to acdc (HRDA)
        source, target = 'cityscapesHR', 'acdcHR'

        # MIC setup
        mask_block_size, mask_ratio = 64, 0.7
        mask_lambda = 0.5
        mask_mode = 'separatetrgaug'

        # AugPatch setup
        aug_mode = 'separatetrgaug'
        aug_lambda = 1.0
        aug_block_size = 64
        num_diff_aug = 16


        for loss_adjustment, geometric_perturb, cls_mask in [
            (False,          True,              'Random'),
            (True,           True,              'Random'),

            (True,           True,              False),
            (True,           False,             'Random'),

            (False,          True,              False),
            (True,           False,             False),

            (False,          False,             False),
            (False,          False,             'Random'),
        ]:
            for seed in seeds:
                gpu_model = 'NVIDIAA40'
                # balance lambda
                aug_lambda = 0.5 if loss_adjustment else 1.0
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = 'v2' if 'cityscapes' in target else False
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # DAFormer + MIC + AugPatch + Self-voting, cityscapes to acdc
    # -------------------------------------------------------------------------
    elif id == 92:
        seeds = [2]
        #        opt,     lr,      schedule,     pmult
        sgd   = ('sgd',   0.0025,  'poly10warm', False)
        adamw = ('adamw', 0.00006, 'poly10warm', True)
        #               uda,                  rcs_T, plcrop, opt_hp
        uda_daformer = ('dacs_a999_fdthings', 0.01,  False,   *adamw)

        uda, rcs_T, plcrop, opt, lr, schedule, pmult = uda_daformer

        architecture = 'daformer_sepaspp'
        rcs_min_crop = 0.5
        backbone = 'mitb5'

        gpu_model = 'NVIDIAGeForceRTX2080Ti'
        inference = 'whole'
         
        # cityscapes to acdc (HRDA)
        source, target = 'cityscapes', 'acdc'
        # source, target = 'cityscapes', 'darkzurich'

        # MIC setup
        mask_block_size, mask_ratio = 32, 0.7
        mask_lambda = 0.5
        mask_mode = 'separatetrgaug'

        # AugPatch setup
        aug_mode = 'separatetrgaug'
        aug_lambda = 1.0
        aug_block_size = 32
        num_diff_aug = 16
        augment_setup={'n': 8, 'm': 30}

        # Self-voting setup
        enable_refine = True
        k = 5
        refine_aug = {'n': 4, 'm': 10}
        num_diff_ref_aug = 32
        # 從1500 iter 開始儲存，在2000 iter開始進行refine
        # 開始儲存最新feature的時間點是 
        # start_iters - (max_bank_size / batch size)
        start_iters = 2000
        max_bank_size = 1000
        # start_iters = 20
        # max_bank_size = 200

        for loss_adjustment, geometric_perturb, cls_mask in [
            (False,          False,             'Random'),
            (False,          True,              'Random'),
            (True,           True,              'Random'),

            (True,           True,              False),
            (True,           False,             'Random'),

            (False,          True,              False),
            (True,           False,             False),

            (False,          False,             False),
        ]:
            for seed in seeds:
                gpu_model = 'NVIDIAA40'
                # balance lambda
                aug_lambda = 0.5 if loss_adjustment else 1.0
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = 'v2' if 'cityscapes' in target else False
                plcrop = True # 不知道為什麼設成True比較好
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # 碩論 with HRDA for Different UDA Benchmarks (Table 2)
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 93:
        seeds = [0, 1, 2]
        architecture, backbone = 'hrda1-512-0.1_daformer_sepaspp', 'mitb5'
        uda, rcs_T = 'dacs_a999_fdthings', 0.01
        crop, rcs_min_crop = '1024x1024', 0.5 * (2 ** 2)
        inference = 'slide'

        adamw = ('adamw', 0.00006, 'poly10warm', True)
        uda_hrda =     ('dacs_a999_fdthings', 0.01,  'v2',   *adamw)
        uda, rcs_T, plcrop, opt, lr, schedule, pmult = uda_hrda

        mask_block_size, mask_ratio = 64, 0.7

        # MIC setup
        mask_lambda = 0.5
        mask_mode = 'separatetrgaug'

        # AugPatch setup
        aug_mode = 'separatetrgaug'
        aug_lambda = 1.0
        aug_block_size = 64
        num_diff_aug = 16
        augment_setup = {'n': 4, 'm': 10}
        cls_mask = 'Random'
        geometric_perturb = True

        loss_adjustment = False

        # Self-voting setup
        enable_refine = False
        k = 5
        refine_aug = {'n': 4, 'm': 10}
        num_diff_ref_aug = 32
        # 從1500 iter 開始儲存，在2000 iter開始進行refine
        # 開始儲存最新feature的時間點是 
        # start_iters - (max_bank_size / batch size)
        start_iters = 2000
        # start_iters = 20
        max_bank_size = 1000

        for seed in seeds:
            for mask_mode in [
                # 'separatetrgaug',
                None
            ]:
                for source,          target in [
                    ('gtaHR',        'cityscapesHR'),
                    ('synthiaHR',    'cityscapesHR'),
                    # ('cityscapesHR', 'acdcHR',       'separate'),
                    # ('cityscapesHR', 'darkzurichHR', 'separate'),
                    ('cityscapesHR', 'acdcHR'),
                    ('cityscapesHR', 'darkzurichHR'),
                ]:
                    gpu_model = 'NVIDIATITANRTX'
                    # plcrop is only necessary for Cityscapes as target domains
                    # ACDC and DarkZurich have no rectification artifacts.
                    # plcrop = 'v2' if 'cityscapes' in target else False
                    plcrop = 'v2'
                    cfg = config_from_vars()
                    cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # AugPatch ablation for aug times and block size
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 94:
        seeds = [0, 1, 2]
        source, target = 'cityscapes', 'acdc'
        architecture, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, False

        # 由於 Rare Class 的表現不佳，懷疑是RCS參數不夠好，因此手動調整
        # 越小稀有類的 sample 機率越高，參考DAFormer Table S2.
        # rcs_T = 0.002

        architecture = 'daformer_sepaspp'
        rcs_min_crop = 0.5
        backbone = 'mitb5'

        gpu_model = 'NVIDIAGeForceRTX2080Ti'
        inference = 'whole'

        # MIC setup
        # mask_mode = None
        mask_block_size, mask_ratio = 32, 0.7
        mask_lambda = 1.0
        mask_mode = 'separate'

        # AugPatch setup
        aug_mode = 'separateaug'
        aug_lambda = 1.0
        num_diff_aug = 8
        augment_setup = {'n': 4, 'm': 20}
        geometric_perturb = {
            'perturb_range': (30, 30, 30),
            'perturb_prob': 0.7
        }
        cls_mask = 'Random'
        mixing_cfg = {
            'mode': 'same',
            'mixing_ratio': 0.5,
            'mixing_type': 'cutmix'
        }

        # Self-voting setup
        enable_refine = False

        loss_adjustment = 5

        for seed in seeds:
            for aug_times in [
                64, 
                128
            ]:
                for block_size in [
                    4,
                    8,
                    16,
                    32,
                    64,
                ]:
                    # if block_size == 256 and aug_times > 4:
                    #     continue
                    gpu_model = 'NVIDIARTX2080Ti'
                    # balance lambda
                    # plcrop is only necessary for Cityscapes as target domains
                    # ACDC and DarkZurich have no rectification artifacts.
                    aug_block_size, num_diff_aug = block_size, aug_times
                    cfg = config_from_vars()
                    cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # AugPatch ablation for geometric perturb
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 95:
        seeds = [0, 1, 2]
        source, target = 'cityscapes', 'acdc'
        architecture, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, False

        # 由於 Rare Class 的表現不佳，懷疑是RCS參數不夠好，因此手動調整
        # 越小稀有類的 sample 機率越高，參考DAFormer Table S2.
        # rcs_T = 0.002

        architecture = 'daformer_sepaspp'
        rcs_min_crop = 0.5
        backbone = 'mitb5'

        gpu_model = 'NVIDIAGeForceRTX2080Ti'
        inference = 'whole'

        # MIC setup
        mask_block_size, mask_ratio = 32, 0.7
        mask_lambda = 1.0
        mask_mode = 'separate'

        # AugPatch setup
        aug_mode = 'separateaug'
        aug_lambda = 1.0
        num_diff_aug = 8
        augment_setup = {'n': 4, 'm': 20}
        aug_block_size = 16
        num_diff_aug = 8
        cls_mask = 'Random'

        mixing_cfg = {
            'mode': 'same',
            'mixing_ratio': 0.5,
            'mixing_type': 'cutmix'
        }

        # Self-voting setup
        enable_refine = False

        loss_adjustment = 1.5
        geometric_perturb = {
            'perturb_range': (15, 15, 15),
            'perturb_prob': 0.7
        }
        geometric_perturb = False

        for seed in seeds:
            # gpu_model = 'NVIDIARTX2080Ti'
            # balance lambda
            # plcrop is only necessary for Cityscapes as target domains
            # ACDC and DarkZurich have no rectification artifacts.

            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # 碩論 with HRDA for Different UDA Benchmarks (Table 2)
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 96:
        seeds = [2, 1, 0]
        architecture, backbone = 'hrda1-512-0.1_daformer_sepaspp', 'mitb5'
        uda, rcs_T = 'dacs_a999_fdthings', 0.01
        crop, rcs_min_crop = '1024x1024', 0.5 * (2 ** 2)
        inference = 'slide'

        # MIC setup
        mask_block_size, mask_ratio = 64, 0.7
        mask_lambda = 1.0

        # AugPatch setup
        aug_lambda = 1.0
        aug_block_size = 32
        num_diff_aug = 8
        augment_setup = {'n': 8, 'm': 30}
        cls_mask = 'Random'
        
        mixing_cfg = False
        # mixing_cfg = {
        #     'mode': 'same',
        #     'mixing_ratio': 0.5,
        #     'mixing_type': 'cutmix'
        # }

        geometric_perturb = {
            'perturb_range': (30, 30, 30),
            'perturb_prob': 0.7
        }
        # geometric_perturb = False
        loss_adjustment = 1.5

        # Self-voting setup
        enable_refine = False

        for seed in seeds:
            for source,             target,          mode in [
                ('cityscapesHR',    'acdcHR',        'separateaug'),
                ('gtaHR',           'cityscapesHR',  'separatetrgaug'),
                # ('cityscapesHR',    'darkzurichHR',  'separateaug'),
                # ('synthiaHR',       'cityscapesHR',  'separatetrgaug'),
            ]:
                gpu_model = 'NVIDIATITANRTX'
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = 'v2' if 'cityscapes' in target else False
                aug_mode = mode
                mask_mode = mode if 'cityscapes' in target else mode[:-3]
                geometric_perturb = None if 'acdc' not in target else geometric_perturb
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # 碩論 with HRDA for Different UDA Benchmarks (Table 2)
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 97:
        seeds = [0, 1, 2]
        architecture, backbone = 'hrda1-512-0.1_daformer_sepaspp', 'mitb5'
        uda, rcs_T = 'dacs_a999_fdthings', 0.01
        crop, rcs_min_crop = '1024x1024', 0.5 * (2 ** 2)
        inference = 'slide'

        # MIC setup
        mask_block_size, mask_ratio = 64, 0.7
        mask_lambda = 1.0

        # AugPatch setup
        aug_mode = 'separatetrgaug'
        aug_lambda = 1.0
        aug_block_size = 32
        num_diff_aug = 8
        augment_setup = {'n': 8, 'm': 30}
        cls_mask = 'Random'
        
        mixing_cfg = {
            'mode': 'same',
            'mixing_ratio': 0.5,
            'mixing_type': 'cutmix'
        }

        # geometric_perturb = {
        #     'perturb_range': (30, 30, 30),
        #     'perturb_prob': 0.5
        # }
        geometric_perturb = False
        loss_adjustment = False

        # Self-voting setup
        enable_refine = False

        for seed in seeds:
            for source,             target,          mode in [
                # ('gtaHR',           'cityscapesHR',  'separatetrgaug'),
                ('cityscapesHR',    'acdcHR',        'separateaug'),
                # ('synthia',    'cityscapes'),
                # ('cityscapes', 'darkzurich'),
            ]:
                gpu_model = 'NVIDIATITANRTX'
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = 'v2' if 'cityscapes' in target else False
                aug_mode = mode
                mask_mode = mode if 'cityscapes' in target else mode[:-3]
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # 碩論 with HRDA for Different UDA Benchmarks (Table 2)
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 98:
        seeds = [2, 1, 0]
        architecture, backbone = 'hrda1-512-0.1_daformer_sepaspp', 'mitb5'
        uda, rcs_T = 'dacs_a999_fdthings', 0.01
        crop, rcs_min_crop = '1024x1024', 0.5 * (2 ** 2)
        inference = 'slide'

        # MIC setup
        mask_block_size, mask_ratio = 64, 0.7
        mask_lambda = 1.0

        # AugPatch setup
        aug_lambda = 1.0
        aug_block_size = 32
        num_diff_aug = 16
        augment_setup = {'n': 8, 'm': 30}
        cls_mask = False
        
        mixing_cfg = {
            'mode': 'same',
            'mixing_ratio': 0.5,
            'mixing_type': 'cutmix'
        }

        geometric_perturb = {
            'perturb_range': (30, 30, 30),
            'perturb_prob': 0.7
        }
        # geometric_perturb = False
        loss_adjustment = 1.5

        # Self-voting setup
        enable_refine = False

        for seed in seeds:
            for source,             target,          mode in [
                # ('gtaHR',           'cityscapesHR',  'separatetrgaug'),
                ('cityscapesHR',    'acdcHR',        'separateaug'),
                # ('synthia',    'cityscapes'),
                # ('cityscapes', 'darkzurich'),
            ]:
                gpu_model = 'NVIDIATITANRTX'
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = 'v2' if 'cityscapes' in target else False
                aug_mode = mode
                mask_mode = mode if 'cityscapes' in target else mode[:-3]
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # 碩論 with HRDA for Different UDA Benchmarks (Table 2)
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 99:
        seeds = [0, 1, 2]
        architecture, backbone = 'hrda1-512-0.1_daformer_sepaspp', 'mitb5'
        uda, rcs_T = 'dacs_a999_fdthings', 0.01
        crop, rcs_min_crop = '1024x1024', 0.5 * (2 ** 2)
        inference = 'slide'

        # MIC setup
        mask_block_size, mask_ratio = 64, 0.7
        mask_lambda = 1.0

        # AugPatch setup
        aug_lambda = 1.0
        aug_block_size = 16
        num_diff_aug = 8
        augment_setup = {'n': 8, 'm': 30}
        cls_mask = 'Random'
        
        # mixing_cfg = False
        mixing_cfg = {
            'mode': 'same',
            'mixing_ratio': 0.5,
            'mixing_type': 'cutmix'
        }

        geometric_perturb = {
            'perturb_range': (30, 30, 30),
            'perturb_prob': 0.7
        }
        # geometric_perturb = False
        loss_adjustment = 5

        # Self-voting setup
        enable_refine = False

        for seed in seeds:
            for source,             target,          mode in [
                ('cityscapesHR',    'acdcHR',        'separateaug'),
                # ('gtaHR',           'cityscapesHR',  'separatetrgaug'),
                # ('cityscapesHR',    'darkzurichHR',  'separateaug'),
                # ('synthiaHR',       'cityscapesHR',  'separatetrgaug'),
            ]:
                gpu_model = 'NVIDIATITANRTX'
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = 'v2' if 'cityscapes' in target else False
                aug_mode = mode
                mask_mode = mode if 'cityscapes' in target else mode[:-3]
                # if 'acdc' in target:
                #     geometric_perturb = False
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # 碩論 with HRDA for Different UDA Benchmarks (Table 2)
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 100:
        seeds = [2, 1, 0]
        architecture, backbone = 'hrda1-512-0.1_daformer_sepaspp', 'mitb5'
        uda, rcs_T = 'dacs_a999_fdthings', 0.01
        crop, rcs_min_crop = '1024x1024', 0.5 * (2 ** 2)
        inference = 'slide'

        # MIC setup
        mask_block_size, mask_ratio = 64, 0.7
        mask_lambda = 0.5

        # AugPatch setup
        aug_lambda = 0.25
        aug_block_size = 16
        num_diff_aug = 8
        augment_setup = {'n': 4, 'm': 30}
        cls_mask = 'Random'
        
        # mixing_cfg = False
        mixing_cfg = {
            'mode': 'same',
            'mixing_ratio': 0.5,
            'mixing_type': 'cutmix'
        }

        geometric_perturb = {
            'perturb_range': (30, 30, 30),
            'perturb_prob': 0.7
        }
        # geometric_perturb = False
        loss_adjustment = 5
        consis_mode = 'unify'

        # Self-voting setup
        enable_refine = False

        for seed in seeds:
            for source,             target,          mode in [
                ('cityscapesHR',    'acdcHR',        'separateaug'),
                # ('cityscapesHR',    'darkzurichHR',  'separateaug'),
                # ('cityscapesHR',    'foggyzurichHR', 'separateaug'),
            ]:
                gpu_model = 'NVIDIATITANRTX'
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = 'v2' if ('foggyzurich' in target) else False
                aug_mode = mode
                mask_mode = mode if 'cityscapes' in target else mode[:-3]
                # mask_mode = None
                # if 'acdc' in target:
                #     geometric_perturb = False
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # 碩論 with HRDA for Different UDA Benchmarks (Table 2)
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 101:
        seeds = [2, 1, 0]
        architecture, backbone = 'hrda1-512-0.1_daformer_sepaspp', 'mitb5'
        uda, rcs_T = 'dacs_a999_fdthings', 0.01
        crop, rcs_min_crop = '1024x1024', 0.5 * (2 ** 2)
        inference = 'slide'

        # MIC setup
        mask_block_size, mask_ratio = 64, 0.7
        mask_lambda = 1.0

        # AugPatch setup
        aug_lambda = 1.0
        aug_block_size = 16
        num_diff_aug = 16
        augment_setup = {'n': 4, 'm': 15}
        cls_mask = 'Random'
        
        mixing_cfg = False
        # mixing_cfg = {
        #     'mode': 'same',
        #     'mixing_ratio': 0.5,
        #     'mixing_type': 'cutmix'
        # }

        gp = {
            'perturb_range': (30, 30, 30),
            'patch_p': 0.7,
            'image_p': 0.5
        }
        # geometric_perturb = False
        loss_adjustment = 5

        # Self-voting setup
        enable_refine = False

        for seed in seeds:
            for source,             target,          mode in [
                # ('cityscapesHR',    'foggyzurichHR',        'separatetrgaug'),
                ('cityscapesHR',    'acdcHR',        'separateaug'),
                # ('cityscapesHR',    'darkzurichHR',  'separateaug'),
                # ('cityscapesHR',    'darkzurichHR',  'separateaug'),
            ]:
                gpu_model = 'NVIDIATITANRTX'
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = 'v2' if ('foggyzurich' in target) else False
                aug_mode = mode
                mask_mode = mode if 'cityscapes' in target else mode[:-3]
                # if 'acdc' in target:
                #     geometric_perturb = False
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # AugPatch ablation for geometric perturb
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 102:
        seeds = [2, 1, 0]
        source, target = 'cityscapes', 'acdc'
        architecture, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, False

        # 由於 Rare Class 的表現不佳，懷疑是RCS參數不夠好，因此手動調整
        # 越小稀有類的 sample 機率越高，參考DAFormer Table S2.
        # rcs_T = 0.002

        architecture = 'daformer_sepaspp'
        rcs_min_crop = 0.5
        backbone = 'mitb5'

        gpu_model = 'NVIDIAGeForceRTX2080Ti'
        inference = 'whole'

        # MIC setup
        mask_block_size, mask_ratio = 32, 0.7
        mask_lambda = 0.5
        mask_mode = 'separate'

        # AugPatch setup
        aug_mode = 'separateaug'
        aug_lambda = 0.25
        num_diff_aug = 8
        augment_setup = {'n': 8, 'm': 30}
        aug_block_size = 16

        mixing_cfg = {
            'mode': 'same',
            'mixing_ratio': 0.5,
            'mixing_type': 'cutmix'
        }
        cls_mask = 'Random'
        geometric_perturb = {
            'perturb_range': (30, 30, 30),
            'patch_p': 0.7,
            'image_p': 0.5
        }

        loss_adjustment = 5

        consis_mode = 'unify'
        
        for seed in seeds:
            # gpu_model = 'NVIDIARTX2080Ti'
            # balance lambda
            # plcrop is only necessary for Cityscapes as target domains
            # ACDC and DarkZurich have no rectification artifacts.
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # AugPatch ablation for geometric perturb
    # -------------------------------------------------------------------------
    # yapf: disable
    # -------------------------------------------------------------------------
    # 碩論 with HRDA for Different UDA Benchmarks (Table 2)
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 103:
        seeds = [2, 1, 0]
        architecture, backbone = 'hrda1-512-0.1_daformer_sepaspp', 'mitb5'
        uda, rcs_T = 'dacs_a999_fdthings', 0.01
        crop, rcs_min_crop = '1024x1024', 0.5 * (2 ** 2)
        inference = 'slide'

        # MIC setup
        mask_block_size, mask_ratio = 64, 0.7
        mask_lambda = 1.0

        # AugPatch setup
        aug_lambda = 0.5
        aug_block_size = 16
        num_diff_aug = 8
        augment_setup = {'n': 8, 'm': 30}
        cls_mask = 'Random'
        
        # mixing_cfg = False
        mixing_cfg = {
            'mode': 'same',
            'mixing_ratio': 0.5,
            'mixing_type': 'cutmix'
        }

        geometric_perturb = {
            'perturb_range': (30, 30, 30),
            'patch_p': 0.5,
            'image_p': 0.5
        }

        consis_mode = 'unify'
        loss_adjustment = 5.0

        for seed in seeds:
            for source,             target,          mode in [
                # ('cityscapesHR',    'acdcHR',        'separateaug'),
                ('cityscapesHR',    'darkzurichHR',  'separateaug'),
                # ('synthiaHR',       'cityscapesHR',  'separatetrgaug'),
            ]:
                gpu_model = 'NVIDIATITANRTX'
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = 'v2' if 'cityscapes' in target else False
                aug_mode = mode
                mask_mode = mode if 'cityscapes' in target else mode[:-3]

                cfg = config_from_vars()
                cfgs.append(cfg)
    elif id == 0:
        seeds = [2]
        source, target = 'cityscapes', 'acdc'
        architecture, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, False

        iters = 1001

        # 由於 Rare Class 的表現不佳，懷疑是RCS參數不夠好，因此手動調整
        # 越小稀有類的 sample 機率越高，參考DAFormer Table S2.
        # rcs_T = 0.002

        architecture = 'daformer_sepaspp'
        rcs_min_crop = 0.5
        backbone = 'mitb5'

        gpu_model = 'NVIDIAGeForceRTX2080Ti'
        inference = 'whole'

        # MIC setup
        mask_block_size, mask_ratio = 32, 0.7
        mask_lambda = 1.0
        mask_mode = 'separate'

        # AugPatch setup
        aug_mode = 'separateaug'
        aug_lambda = 1.0
        num_diff_aug = 8
        augment_setup = {'n': 2, 'm': 10}
        aug_block_size = 128

        cls_mask = False
        geometric_perturb = False
        mixing_cfg = False

        # Self-voting setup
        enable_refine = False

        loss_adjustment = 5

        mix = {
            'mode': 'same',
            'mixing_ratio': 0.5,
            'mixing_type': 'cutmix'
        }
        gp = {
            'perturb_range': (30, 30, 30),
            'patch_p': 0.7,
            'image_p': 0.5
        }
        consis_mode = False
        
        for seed in seeds:
            # gpu_model = 'NVIDIARTX2080Ti'
            # balance lambda
            # plcrop is only necessary for Cityscapes as target domains
            # ACDC and DarkZurich have no rectification artifacts.
            for enable_mixing, enable_gp, enable_clsMask in [
                # (False, True, True),
                # (True, False, True),
                # (True, True, False),

                # (True, False, False),
                # (False, True, False),
                # (False, False, True),

                # (False, False, False),
                # (True, False, False),
                (True, True, False),
                (True, True, True)

                # (True, True, True),
            ]:
                mixing_cfg = mix if enable_mixing else False
                geometric_perturb = gp if enable_gp else False
                cls_mask = 'Random' if enable_clsMask else False

                cfg = config_from_vars()
                cfgs.append(cfg)
    else:
        raise NotImplementedError('Unknown id {}'.format(id))

    return cfgs
