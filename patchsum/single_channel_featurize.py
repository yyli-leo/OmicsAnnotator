import numpy as np
import cv2
import torch
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights
)
import torch.nn as nn


class SingleChannelFeaturize(object):
    def __init__(self, **kwargs):
        pass

    def _preprocess(self, img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().data.numpy()
        return img

    def __call__(self, img):
        raise NotImplementedError

    def featurize(self, img):
        if len(img.shape) == 2:
            return self(img)
        elif len(img.shape) == 3:
            # Assuming CHW order
            feats = []
            for channel_im in img:
                feats.append(self(channel_im))
            return np.concatenate(feats)


class SingleChannelRaw(SingleChannelFeaturize):
    def __call__(self, img):
        return img.flatten()


class SingleChannelAverage(SingleChannelFeaturize):
    def __call__(self, img):
        val = np.nanmean(img.flatten(), keepdims=True)
        return val


class SingleChannelMedian(SingleChannelFeaturize):
    def __call__(self, img):
        val = np.nanmedian(img.flatten(), keepdims=True)
        return val


class SingleChannelHistogram(SingleChannelFeaturize):
    def __init__(self, bins=np.arange(0, 256, 8), **kwargs):
        self.bins = bins
        SingleChannelFeaturize.__init__(self, **kwargs)

    def _preprocess(self, img):
        _img = super()._preprocess(img)
        if _img.dtype in [np.dtype(int), np.dtype('uint8')]:
            _img = np.clip(_img, 0, 255).astype('uint8')
        elif _img.dtype in [np.dtype(float)]:
            _img = (np.clip(_img, 0, 1) * 255).astype('uint8')
        else:
            raise TypeError
        return _img

    def __call__(self, img):
        _img = self._preprocess(img)
        counts, _ = np.histogram(_img, bins=self.bins)
        freq = counts / img.size
        assert len(freq) == len(self.bins) - 1
        return freq


class SingleChannelSIFT(SingleChannelFeaturize):
    def __init__(self, **kwargs):
        self.featurizer = cv2.SIFT_create()
        SingleChannelFeaturize.__init__(self, **kwargs)

    def _preprocess(self, img):
        _img = super()._preprocess(img)
        _img = (np.clip(_img, 0, 1) * 255).astype('uint8')
        _img = cv2.medianBlur(_img, 5)
        return _img

    def __call__(self, img):
        _img = self._preprocess(img)
        _, descriptors = self.featurizer.detectAndCompute(_img, None)
        feat = descriptors.mean(0)
        return feat


class SingleChannelORB(SingleChannelSIFT):
    def __init__(self, **kwargs):
        self.featurizer = cv2.ORB_create()
        SingleChannelFeaturize.__init__(self, **kwargs)


class SingleChannelImageNetFeature(SingleChannelFeaturize):
    def __init__(self, method='resnet50', **kwargs):
        if method == 'resnet50':
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
            self.featurizer = nn.Sequential(*list(model.children())[:-1])
            self.featurizer.eval()
            self.model_preprocess = weights.transforms()
        elif method == 'efficientnet_v2_s':
            weights = EfficientNet_V2_S_Weights.DEFAULT
            model = efficientnet_v2_s(weights=weights)
            self.featurizer = nn.Sequential(*list(model.children())[:-1])
            self.featurizer.eval()
            self.model_preprocess = weights.transforms()
        else:
            raise ValueError("Method %s not supported" % method)
        SingleChannelFeaturize.__init__(self, **kwargs)

    def _preprocess(self, img):
        if isinstance(img, np.ndarray):
            img = torch.tensor(img)
        if torch.is_floating_point(img):
            img = (torch.clip(img, 0, 1) * 255).byte()
        elif img.dtype == torch.uint8:
            pass
        else:
            raise TypeError("Cannot handle type %s" % str(img.dtype))
        img = img.unsqueeze(0).unsqueeze(0)
        img = torch.cat([img] * 3, 1)
        img = self.model_preprocess(img)
        return img

    def __call__(self, img):
        _img = self._preprocess(img)
        emb = self.featurizer(_img)  # Output shaped as (1, 2048, 1, 1)
        emb = emb.squeeze().data.numpy()
        return emb


class SingleChannelConvNetRandomFeature(SingleChannelImageNetFeature):
    def __init__(self, **kwargs):
        model = resnet50(weights=None)
        self.featurizer = nn.Sequential(*list(model.children())[:-1])
        self.featurizer.apply(self.weights_init)
        self.featurizer.eval()
        self.model_preprocess = ResNet50_Weights.DEFAULT.transforms()
        SingleChannelFeaturize.__init__(self, **kwargs)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
