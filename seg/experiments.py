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
                'semantic_mixing': semantic_mixing,
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

    # -------------------------------------------------------------------------
    # AugPatch with DAFormer for Different UDA Benchmarks (Table 2)
    # -------------------------------------------------------------------------
    # yapf: disable
    if id == 92:
        seeds = [2, 1, 0]
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
        aug_block_size = 16
        semantic_mixing = {'mixing_type': 'cutmix'}
        cls_mask = 'Random'
        geometric_perturb = {
            'perturb_range': (30, 30, 30),
            'patch_p': 0.7,
            'image_p': 0.5
        }

        loss_adjustment = 5

        consis_mode = 'unify'
        
        for seed in seeds:
            for source,         target in [
                ('cityscapes',  'acdc'),
                ('cityscapes',  'darkzurich')
            ]:
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # AugPatch with HRDA for Different UDA Benchmarks (Table 2)
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 93:
        seeds = [2, 1, 0]
    
        # Backbone setup
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
        semantic_mixing = {'mixing_type': 'classmix'}
        geometric_perturb = {
            'perturb_range': (30, 30, 30),
            'patch_p': 0.5,
            'image_p': 0.5
        }

        loss_adjustment = 5.0
    
        # consis_mode = 'unify'
        consis_mode = False

        for seed in seeds:
            for source,             target,          mode in [
                # ('cityscapesHR',    'acdcHR',        'separateaug'),
                ('cityscapesHR',    'darkzurichHR',  'separateaug'),
                ('cityscapesHR',    'foggyzurichHR',  'separateaug'),
            ]:
                gpu_model = 'NVIDIATITANRTX'
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                plcrop = 'v2' if 'foggyzurich' in target else False
                aug_mode = mode
                mask_mode = mode if 'cityscapes' in target else mode[:-3]

                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Ablation study with component analyze
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 94:
        # Basic setup
        seeds = [0, 1, 2]
        source, target = 'cityscapes', 'acdc'
        gpu_model = 'NVIDIAGeForceRTX2080Ti'

        # Backbone setup
        architecture, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, False

        architecture = 'daformer_sepaspp'
        rcs_min_crop = 0.5
        backbone = 'mitb5'

        inference = 'whole'

        # MIC setup
        mask_block_size, mask_ratio = 32, 0.7
        mask_lambda = 1.0
        mask_mode = None

        # AugPatch setup
        aug_mode = 'separateaug'
        aug_lambda = 1.0
        num_diff_aug = 8
        aug_block_size = 16
        augment_setup = {'n': 8, 'm': 30}

        # consistency setup
        consis_mode = False
        loss_adjustment = False

        for seed in seeds:
            for mixing, geometric, class_masking in [
                (False, False,     False),

                (True,  False,     False),
                (False, True,      False),
                (False, False,     True),

                (False, True,      True),
                (True,  False,     True),
                (True,  True,      False),

                (True,  True,      True),
            ]:
                # balance lambda
                # plcrop is only necessary for Cityscapes as target domains
                # ACDC and DarkZurich have no rectification artifacts.
                semantic_mixing = {'mixing_type': 'classmix'} if mixing else False
                geometric_perturb = {
                    'perturb_range': (30, 30, 30),
                    'image_p': 0.5,
                    'patch_p': 0.7
                } if geometric else False
                cls_mask = 'Random' if class_masking else False

                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # AugPatch ablation for aug times and block size
    # -------------------------------------------------------------------------
    # yapf: disable
    elif id == 95:
        # Basic setup
        seeds = [0, 1, 2]
        source, target = 'cityscapes', 'acdc'
        gpu_model = 'NVIDIAGeForceRTX2080Ti'

        # Backbone setup
        architecture, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, False

        architecture = 'daformer_sepaspp'
        rcs_min_crop = 0.5
        backbone = 'mitb5'

        inference = 'whole'

        # MIC setup
        mask_block_size, mask_ratio = 32, 0.7
        mask_lambda = 1.0
        mask_mode = None

        # AugPatch setup
        aug_mode = 'separateaug'
        aug_lambda = 1.0
        num_diff_aug = 8
        augment_setup = {'n': 8, 'm': 30}
        geometric_perturb = False
        cls_mask = 'Random'
        semantic_mixing = False

        # consistency setup
        consis_mode = False
        loss_adjustment = 5

        for seed in seeds:
            for aug_times, block_size in [
                (16, 256),
                (32, 256),
                (32, 128),
                (64, 256),
                (64, 128),
                (64, 64),
                (64, 2),
                (64, 1), 
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
    elif id == 96:
        # Basic setup
        seeds = [0, 1, 2]
        source, target = 'cityscapes', 'acdc'
        gpu_model = 'NVIDIAGeForceRTX2080Ti'

        # Backbone setup
        architecture, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, False

        architecture = 'daformer_sepaspp'
        rcs_min_crop = 0.5
        backbone = 'mitb5'

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

        semantic_mixing = False

        loss_adjustment = 5

        image_p = 0.5
        for seed in seeds:
            for perturb_range, patch_p in [
                ((15, 15, 15), 0.3),
                ((15, 15, 15), 0.5),
                ((15, 15, 15), 0.7),
                ((30, 30, 30), 0.3),
                ((30, 30, 30), 0.5),
                ((30, 30, 30), 0.7),
                ((45, 45, 45), 0.3),
                ((45, 45, 45), 0.5),
                ((45, 45, 45), 0.7),
            ]:
                geometric_perturb = {
                    'perturb_range': perturb_range,
                    'image_p': image_p,
                    'patch_p': patch_p
                }
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
        semantic_mixing = False

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
                semantic_mixing = mix if enable_mixing else False
                geometric_perturb = gp if enable_gp else False
                cls_mask = 'Random' if enable_clsMask else False

                cfg = config_from_vars()
                cfgs.append(cfg)
    else:
        raise NotImplementedError('Unknown id {}'.format(id))

    return cfgs
