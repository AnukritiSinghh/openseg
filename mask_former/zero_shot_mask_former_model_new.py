# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms.functional as fn

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from .modeling.clip_adapter import (
    ClipAdapter,
    MaskFormerClipAdapter,
    build_prompt_learner,
)
from .mask_former_model import MaskFormer

from .modeling.clip_adapter.utils import build_clip_model, crop_with_mask, CLIP
from .modeling.clip_adapter.text_prompt import PromptExtractor


@META_ARCH_REGISTRY.register()
class ZeroShotMaskFormer(MaskFormer):
    """
    Main class for zero shot mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        clip_model_name: str,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        clip_adapter: nn.Module,
        region_clip_adapter: nn.Module = None,
        criterion: nn.Module,
        num_queries: int,
        panoptic_on: bool,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        clip_ensemble: bool,
        clip_ensemble_weight: float,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        conditionallearnable=False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            clip_adapter: adapter for clip-based mask classification
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            panoptic_on=panoptic_on,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        self.clip_adapter: ClipAdapter = clip_adapter
        self._region_clip_adapter = region_clip_adapter
        
        self.clip_model = build_clip_model(clip_model_name)
        self.image_encoder = self.clip_model.visual
        #print(image_encoder.shape,"image_encoder")
        self.dtype = self.clip_model.dtype
        
        self.conditionallearnable = conditionallearnable

        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight
        
       

    @classmethod
    def from_config(cls, cfg):
        init_kwargs = MaskFormer.from_config(cfg)
        prompt_learner = build_prompt_learner(cfg.MODEL.CLIP_ADAPTER)
        region_clip_adapter = None
        if cfg.MODEL.CLIP_ADAPTER.SEPERATE_ADAPTER:
            log_first_n(
                logging.WARNING,
                "Using different head for region classification and query classification",
            )
            cls_prompt_learner = build_prompt_learner(
                cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER
            )
            region_clip_adapter = MaskFormerClipAdapter(
                cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.CLIP_MODEL_NAME,
                cls_prompt_learner,
                mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
                mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
                mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
                mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
                region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
            )

        clip_adapter = MaskFormerClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            prompt_learner,
            mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
            mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
            mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
            mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
            region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
        )
        init_kwargs["clip_adapter"] = clip_adapter
        init_kwargs["region_clip_adapter"] = region_clip_adapter
        init_kwargs["clip_ensemble"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        init_kwargs[
            "clip_ensemble_weight"
        ] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT
        init_kwargs["clip_model_name"] = cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME
        init_kwargs["conditionallearnable"] = cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER    #changed
        return init_kwargs

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        
        dataset_name = [x["meta"]["dataset_name"] for x in batched_inputs]
        assert len(set(dataset_name)) == 1
        dataset_name = dataset_name[0]
        
        #print(type(batched_inputs),"batched_inputs")
        
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        
        #image = [x["image"].to(self.device) for x in batched_inputs]
        image = images.tensor
        image2 = images.tensor
        # image2 = fn.resize(image2, size=[640,640])
        #print(image2.shape,"image shape 1")      #[batch_size, 3, 640, 640]
        #print(image.shape,"image shape")   #[12, 640, 640]  should be [batch_size, 3, 244, 224] [batch_size, 3, 640, 640]
    
       

        features = self.backbone(images.tensor)    
        outputs = self.sem_seg_head(features)
        class_names = self.get_class_name_list(dataset_name)
        
        
        features_unmasked = self.image_encoder(image.type(self.dtype))
        features_unmasked = features_unmasked / features_unmasked.norm(dim=-1, keepdim=True)
        #exit()

        
        if self.conditionallearnable:                                             
            text_features = self.clip_adapter.get_text_features(class_names, features_unmasked) 
            
        else:
            text_features = self.clip_adapter.get_text_features(class_names, features_unmasked) 
            
        outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
            text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
        )
        if self.training:
            if "aux_outputs" in outputs.keys():
                for i in range(len(outputs["aux_outputs"])):
                    outputs["aux_outputs"][i][
                        "pred_logits"
                    ] = self.clip_adapter.get_sim_logits(
                        text_features,
                        self.clip_adapter.normalize_feature(
                            outputs["aux_outputs"][i]["pred_logits"]
                        ),
                    )
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = image_size[0]
                width = image_size[1]
                mask_pred_result = sem_seg_postprocess(
                    mask_pred_result, image_size, height, width
                )
                image = input_per_image["image"].to(self.device)
                # semantic segmentation inference
                r = self.semantic_inference(
                    mask_cls_result, mask_pred_result, image, image2, class_names, dataset_name, features_unmasked,
                )
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = sem_seg_postprocess(r, image_size, height, width)
                processed_results.append({"sem_seg": r})

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = self.panoptic_inference(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

            return processed_results

    def semantic_inference(self, mask_cls, mask_pred, image, image2, class_names, dataset_name, features_unmasked):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]    #[100,171] should be [99,172]
        mask_pred = mask_pred.sigmoid()
        base_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 27, 28, 31, 32, 35, 36, 37, 38, 39, 40, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 146, 147, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182]
        if self.clip_ensemble:
            clip_cls, valid_flag = self.region_clip_adapter(
                image, class_names, mask_pred, image2, normalize=True, features_unmasked=features_unmasked,
            )
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
            else:
                clip_cls = clip_cls[0]
            # softmax before index or after?
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                map_back_clip_cls[valid_flag] = clip_cls
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor(
                        MetadataCatalog.get(dataset_name).trainable_flag
                    ).to(mask_cls.device)[None, :]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)
                mask_cls = trained_mask * torch.pow(
                    mask_cls, self.clip_ensemble_weight
                ) * torch.pow(map_back_clip_cls, 1 - self.clip_ensemble_weight) + (
                    1 - trained_mask
                ) * torch.pow(
                    mask_cls, 1 - self.clip_ensemble_weight
                ) * torch.pow(
                    map_back_clip_cls, self.clip_ensemble_weight
                )
            else:
                # only clip model predictions are used
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        #print(semseg.shape,"semseg")
        #exit()
        # offset = torch.zeros(semseg.shape).to(semseg.device)        # [171, 640, 962]
        #print(offset.shape,"offset")
        #exit()
        # offset[base_class_index,:,:] = 0.5
        #print(offset,"offset")
        #exit()
        # semseg = semseg - offset
        return semseg

    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        #print(class_names,"class_names")
        return class_names

    @property
    def region_clip_adapter(self):
        if self._region_clip_adapter is None:
            return self.clip_adapter
        return self._region_clip_adapter
    
    def get_image_features(self, image: torch.Tensor):
        image_features = self.clip_model.visual(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #print(image_features.shape,"image_features inside image_feat")
        return image_features
    
    
    def _preprocess_image(
        self, image: torch.Tensor, mask: torch.Tensor, normalize: bool = True
    ):
        """crop, mask and normalize the image

        Args:
            image ([type]): [C,H,W]
            mask ([type]): [K,H,W
            normalize (bool, optional): [description]. Defaults to True.
        """
        dtype = mask.dtype
        bin_mask = mask > self.mask_thr
        valid = bin_mask.sum(dim=(-1, -2)) > 0
        bin_mask = bin_mask[valid]
        mask = mask[valid]
        if not self.mask_matting:
            mask = bin_mask
        bin_mask = BitMasks(bin_mask)
        bboxes = bin_mask.get_bounding_boxes()
        # crop,mask
        regions = [
            crop_with_mask(
                image.type(dtype),
                single_mask.type(dtype),
                bbox,
                fill=self.mask_fill,
                expand_ratio=self.mask_expand_ratio,
            )[None, ...]
            for bbox, single_mask in zip(bboxes, mask)
        ]
        if len(regions) == 0:
            return None, valid
        if normalize:
            regions = [(r - self.pixel_mean) / self.pixel_std for r in regions]
        # resize
        if self.region_resized:
            regions = [
                F.interpolate(r, size=(224, 224), mode="bicubic") for r in regions
            ]
            regions = torch.cat(regions)
        return regions, valid
    
