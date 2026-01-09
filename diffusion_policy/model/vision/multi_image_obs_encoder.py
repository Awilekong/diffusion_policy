from typing import Dict, Tuple, Union, Callable, Optional
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class FixedCrop(nn.Module):
    """固定位置的裁剪，用于自定义 crop 区域"""
    def __init__(self, top: int, left: int, height: int, width: int):
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, img):
        return TF.crop(img, self.top, self.left, self.height, self.width)

    def __repr__(self):
        return f"FixedCrop(top={self.top}, left={self.left}, height={self.height}, width={self.width})"


class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            # crop_pos: [top, left] 指定裁剪起始位置，支持 tuple 或 dict 格式
            # 如果为 None，则根据 random_crop 决定使用随机裁剪或中心裁剪
            crop_pos: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            # use state (low-dim) input along with image observations
            use_state_input: bool=True
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        # 调试回调函数
        self.debug_callback: Optional[Callable] = None

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure cropper
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    # 解析 crop_shape
                    if isinstance(crop_shape, dict):
                        crop_h, crop_w = crop_shape[key]
                    else:
                        crop_h, crop_w = crop_shape

                    # 解析 crop_pos
                    this_crop_pos = None
                    if crop_pos is not None:
                        if isinstance(crop_pos, dict):
                            this_crop_pos = crop_pos.get(key, None)
                        else:
                            this_crop_pos = crop_pos

                    # 根据配置选择裁剪方式
                    if this_crop_pos is not None:
                        # 使用自定义位置的固定裁剪
                        top, left = this_crop_pos
                        this_randomizer = FixedCrop(
                            top=top, left=left, height=crop_h, width=crop_w
                        )
                    elif random_crop:
                        # 训练时随机裁剪，推理时中心裁剪
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=crop_h,
                            crop_width=crop_w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        # 中心裁剪
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(crop_h, crop_w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.use_state_input = use_state_input

    def forward(self, obs_dict):
        batch_size = None
        features = list()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            transformed_imgs = {}  # 用于调试回调
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)

                # 保存 transform 后的图像用于调试
                transformed_imgs[key] = img

                feature = self.key_model_map[key](img)
                features.append(feature)

            # 调试回调：记录 transform 后送入 ResNet 的图像（Stage 4）
            if self.debug_callback is not None and len(transformed_imgs) > 0:
                self.debug_callback('stage4_final_to_unet', transformed_imgs)

        # process lowdim input (conditionally based on use_state_input)
        if self.use_state_input:
            for key in self.low_dim_keys:
                data = obs_dict[key]
                if batch_size is None:
                    batch_size = data.shape[0]
                else:
                    assert batch_size == data.shape[0]
                assert data.shape[1:] == self.key_shape_map[key]
                features.append(data)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape
