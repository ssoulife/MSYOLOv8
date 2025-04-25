# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .MobileNetV4 import *
# (
#     MobileNetV4ConvSmall,
#     MobileNetV4ConvMedium,
#     MobileNetV4ConvLarge,
#     MobileNetV4HybridMedium,
#     MobileNetV4HybridLarge,
# )


from .shufflenet2 import (
    conv_bn_relu_maxpool,
    Shuffle_Block,
    Shuffle_Block_single,
)

from .shufflenetV2orgin import (
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)


from .shufflenetV2 import (
    Conv_maxpool,
    ShuffleNetV2,
)


from .efficientnet import (
    efficient,
)


__all__ = (
    "MobileNetV4ConvSmall",
    "MobileNetV4ConvMedium",
    "MobileNetV4ConvLarge",
    "MobileNetV4HybridMedium",
    "MobileNetV4HybridLarge",
    "conv_bn_relu_maxpool",
    "Shuffle_Block",
    "Shuffle_Block_single",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
    "Conv_maxpool",
    "ShuffleNetV2",
    'efficient',
)
