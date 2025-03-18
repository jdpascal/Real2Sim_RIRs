"""Layers used for up-sampling or down-sampling images.

Many functions are ported from https://github.com/NVlabs/stylegan2.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from op import upfirdn2d


# Function ported from StyleGAN2
def get_weight(module,
               shape,
               weight_var='weight',
               kernel_init=None):
  """Get/create weight tensor for a convolution or fully-connected layer."""

  return module.param(weight_var, kernel_init, shape)


class Conv2d(nn.Module):
  """Conv2d layer with optimal upsampling and downsampling (StyleGAN2)."""

  def __init__(self, in_ch, out_ch, kernel, up=False, down=False,
               resample_kernel=(1, 3, 3, 1),
               use_bias=True,
               kernel_init=None):
    super().__init__()
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
    if kernel_init is not None:
      self.weight.data = kernel_init(self.weight.data.shape)
    if use_bias:
      self.bias = nn.Parameter(torch.zeros(out_ch))

    self.up = up
    self.down = down
    self.resample_kernel = resample_kernel
    self.kernel = kernel
    self.use_bias = use_bias

  def forward(self, x):
    if self.up:
      x = upsample_conv_2d_first_dim(x, self.weight, k=self.resample_kernel)
    elif self.down:
      x = conv_downsample_2d_first_dim(x, self.weight, k=self.resample_kernel)
    else:
      x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)

    if self.use_bias:
      x = x + self.bias.reshape(1, -1, 1, 1)

    return x


def naive_upsample_2d(x, factor=2):
  _N, C, H, W = x.shape
  x = torch.reshape(x, (-1, C, H, 1, W, 1))
  x = x.repeat(1, 1, 1, factor, 1, factor)
  return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x, factor=2):
  _N, C, H, W = x.shape
  x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
  return torch.mean(x, dim=(3, 5))


def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
  """Fused `upsample_2d()` followed by `tf.nn.conv2d()`.

     Padding is performed only once at the beginning, not between the
     operations.
     The fused op is considerably more efficient than performing the same
     calculation
     using standard TensorFlow ops. It supports gradients of arbitrary order.
     Args:
       x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
         C]`.
       w:            Weight tensor of the shape `[filterH, filterW, inChannels,
         outChannels]`. Grouped convolution can be performed by `inChannels =
         x.shape[0] // numGroups`.
       k:            FIR filter of the shape `[firH, firW]` or `[firN]`
         (separable). The default is `[1] * factor`, which corresponds to
         nearest-neighbor upsampling.
       factor:       Integer upsampling factor (default: 2).
       gain:         Scaling factor for signal magnitude (default: 1.0).

     Returns:
       Tensor of the shape `[N, C, H * factor, W * factor]` or
       `[N, H * factor, W * factor, C]`, and same datatype as `x`.
  """

  assert isinstance(factor, int) and factor >= 1

  # Check weight shape.
  assert len(w.shape) == 4
  convH = w.shape[2]
  convW = w.shape[3]
  inC = w.shape[1]
  outC = w.shape[0]

  assert convW == convH

  # Setup filter kernel.
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * (gain * (factor ** 2))
  p = (k.shape[0] - factor) - (convW - 1)

  stride = (factor, factor)

  # Determine data dimensions.
  stride = [1, 1, factor, factor]
  output_shape = ((_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW)
  output_padding = (output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH,
                    output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW)
  assert output_padding[0] >= 0 and output_padding[1] >= 0
  num_groups = _shape(x, 1) // inC

  # Transpose weights.
  w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
  w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
  w = torch.reshape(w, (num_groups * inC, -1, convH, convW))

  x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)
  ## Original TF code.
  # x = tf.nn.conv2d_transpose(
  #     x,
  #     w,
  #     output_shape=output_shape,
  #     strides=stride,
  #     padding='VALID',
  #     data_format=data_format)
  ## JAX equivalent

  return upfirdn2d(x, torch.tensor(k),
                   pad=((p + 1) // 2 + factor - 1, p // 2 + 1))

def upsample_conv_2d_first_dim(x, w, k=None, factor=2, gain=1):
  """Fused `upsample_2d()` followed by `tf.nn.conv2d()`.

     Padding is performed only once at the beginning, not between the
     operations.
     The fused op is considerably more efficient than performing the same
     calculation
     using standard TensorFlow ops. It supports gradients of arbitrary order.
     Args:
       x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
         C]`.
       w:            Weight tensor of the shape `[filterH, filterW, inChannels,
         outChannels]`. Grouped convolution can be performed by `inChannels =
         x.shape[0] // numGroups`.
       k:            FIR filter of the shape `[firH, firW]` or `[firN]`
         (separable). The default is `[1] * factor`, which corresponds to
         nearest-neighbor upsampling.
       factor:       Integer VERTICAL upsampling factor (default: 2).
       gain:         Scaling factor for signal magnitude (default: 1.0).

     Returns:
       Tensor of the shape `[N, C, H * factor, W * 1]` or
       `[N, H * factor, W * 1, C]`, and same datatype as `x`.
  """

  assert isinstance(factor, int) and factor >= 1

  # Check weight shape.
  assert len(w.shape) == 4
  outC, inC, convH, convW = w.shape

    # On s'attend à convW=1 si on veut agir uniquement verticalement
    # (sinon on peut laisser convW > 1, mais alors on fera un mini-filtrage horizontal).
    # => ajuster selon le besoin réel.


  # Setup filter kernel.
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * (gain * (factor ** 2))
  p = (k.shape[0] - factor) - (convH - 1)

  stride = (factor, 1)
  # Calcul de la taille de sortie (verticalement * factor)
  H_out = (x.shape[2] - 1) * factor + convH
  W_out = x.shape[3] + (convW - 1)  # convW=1 => W_out = W + 0
  output_shape = (H_out, W_out)

  # Determine data dimensions.
  delta_H = H_out - (x.shape[2] - 1)*stride[0] - convH
  delta_W = W_out - (x.shape[3] - 1)*stride[1] - convW
  output_padding = (delta_H, delta_W)
  assert output_padding[0] >= 0 and output_padding[1] >= 0

  x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)
  # Sauf que StyleGAN2 combine tout en un; 
  # typiquement, si on veut absolument "tout en un", on ferait
  # conv_transpose2d avec stride=(1,1) + upfirdn2d(up=(factor,1)).
  # Mais on peut faire comme ci-dessous pour respecter la structure initiale.

  # p = ... calcul du padding en vertical
  p = (k.shape[0] - factor) - (convH - 1)
  # On center le padding, dimension vertical: ((p+1)//2, p//2 + quelqueChose)
  # dimension horizontale => 0
  pad_vertical_before = (p + 1)//2 + factor - 1
  pad_vertical_after = p//2 + 0
  # pad sous forme (top, bottom, left, right)
  pad = (pad_vertical_before, pad_vertical_after, 0, 0)

  x = upfirdn2d(x, torch.tensor(k), pad=pad)

  return x



def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
  """Fused `tf.nn.conv2d()` followed by `downsample_2d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same
    calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        w:            Weight tensor of the shape `[filterH, filterW, inChannels,
          outChannels]`. Grouped convolution can be performed by `inChannels =
          x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]` or
        `[N, H // factor, W // factor, C]`, and same datatype as `x`.
  """

  assert isinstance(factor, int) and factor >= 1
  _outC, _inC, convH, convW = w.shape
  assert convW == convH
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * gain
  p = (k.shape[0] - factor) + (convW - 1)
  s = [factor, factor]
  x = upfirdn2d(x, torch.tensor(k),
                pad=((p + 1) // 2, p // 2))
  return F.conv2d(x, w, stride=s, padding=0)

def conv_downsample_2d_first_dim(x, w, k=None, factor=2, gain=1):
  """Fused `tf.nn.conv2d()` followed by `downsample_2d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same
    calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        w:            Weight tensor of the shape `[filterH, filterW, inChannels,
          outChannels]`. Grouped convolution can be performed by `inChannels =
          x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer VERTICAL downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // 1]` or
        `[N, H // factor, W // 1, C]`, and same datatype as `x`.
  """

  assert isinstance(factor, int) and factor >= 1
  _outC, _inC, convH, convW = w.shape
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * gain
  # On applique d'abord la conv2d (stride=(1,1)), pour ne pas réduire la largeur
  x = F.conv2d(x, w, stride=1, padding=(convH//2, convW//2))
  p = (k.shape[0] - factor) 
  # On centre le padding, par ex:
  pad_vertical_before = (p + 1)//2
  pad_vertical_after = p//2
  pad = (pad_vertical_before, pad_vertical_after, 0, 0)
  x = upfirdn2d(x, torch.tensor(k), factor=(factor, 1),
                pad=pad)
  return x


def _setup_kernel(k):
  k = np.asarray(k, dtype=np.float32)
  if k.ndim == 1:
    k = np.outer(k, k)
  k /= np.sum(k)
  assert k.ndim == 2
  assert k.shape[0] == k.shape[1]
  return k


def _shape(x, dim):
  return x.shape[dim]


def upsample_2d(x, k=None, factor=2, gain=1):
  r"""Upsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and upsamples each image with the given filter. The filter is normalized so
    that
    if the input pixels are constant, they will be scaled by the specified
    `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded
    with
    zeros so that its shape is a multiple of the upsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          nearest-neighbor upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]`
  """
  assert isinstance(factor, int) and factor >= 1
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * (gain * (factor ** 2))
  p = k.shape[0] - factor
  return upfirdn2d(x, torch.tensor(k),
                   up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))

def upsample_1d(x, k=None, factor=2, gain=1):
    r"""
    Upsample un batch d'images 2D (N, C, H, W) seulement le long de H,
    en utilisant un FIR filter de forme (kH, 1).
    """
    assert isinstance(factor, int) and factor >= 1

    # Si aucun kernel fourni, on en prend un trivial vertical, e.g. [1]*factor
    if k is None:
        k = [1] * factor
    # _setup_kernel crée un kernel 2D normalisé (kH, kW).
    # Ici, k aura shape (kH, kH) si c'était un vecteur 1D. 
    # On veut un kernel (kH, 1). 
    # Pour y arriver, on peut faire une petite manipulation :
    k_arr = np.array(k, dtype=np.float32)
    k_2d = np.reshape(k_arr, (-1, 1))  # (kH, 1)
    k_2d = k_2d / np.sum(k_2d)         # normalisation
    k_2d = k_2d * (gain * (factor**2))

    p = k_2d.shape[0] - factor
    # On va faire un upsample vertical seulement => up=(factor,1)
    # On doit calculer le padding (top, bottom, left, right).
    # Suppose qu'on fait un padding centré : top=(p+1)//2 + factor - 1, bottom=p//2
    top_pad = (p + 1)//2 + factor - 1
    bottom_pad = p//2
    left_pad = 0
    right_pad = 0
    pad = (top_pad, bottom_pad, left_pad, right_pad)

    # On appelle upfirdn2d pour faire l'upsample vertical
    # (plus le filtrage FIR vertical).
    x = upfirdn2d(x, torch.tensor(k_2d),
                  up=(factor, 1), down=(1, 1), pad=pad)
    return x



def downsample_2d(x, k=None, factor=2, gain=1):
  r"""Downsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and downsamples each image with the given filter. The filter is normalized
    so that
    if the input pixels are constant, they will be scaled by the specified
    `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded
    with
    zeros so that its shape is a multiple of the downsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]`
  """

  assert isinstance(factor, int) and factor >= 1
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * gain
  p = k.shape[0] - factor
  return upfirdn2d(x, torch.tensor(k),
                   down=factor, pad=((p + 1) // 2, p // 2))


def downsample_1d(x, k=None, factor=2, gain=1):
    r"""
    Downsample (N, C, H, W) seulement le long de H,
    avec un FIR filter vertical (kH,1).
    """
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k_arr = np.array(k, dtype=np.float32)
    k_2d = np.reshape(k_arr, (-1, 1))  # (kH,1)
    k_2d = k_2d / np.sum(k_2d)
    k_2d = k_2d * gain
    
    p = k_2d.shape[0] - factor
    top_pad = (p + 1)//2
    bottom_pad = p//2
    pad = (top_pad, bottom_pad, 0, 0)

    # Appel upfirdn2d en mode down=(factor,1)
    x = upfirdn2d(x, torch.tensor(k_2d),
                  up=(1,1), down=(factor,1), pad=pad)
    return x
