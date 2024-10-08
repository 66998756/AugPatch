# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add valid_mask_size
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class FoggyZurichDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(FoggyZurichDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)
        self.valid_mask_size = [1080, 1920]
