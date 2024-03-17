from typing import Iterable
from mlx import nn

from yolov9_mlx.models import modules


LayerDefinition = tuple[None | int | Iterable[int], nn.Module]


class Yolov9Factory:
    """Factory class to create layers in Base Yolov9 model.
    """
    @staticmethod
    def create_layers(in_channels: int, num_classes: int) -> tuple[list[LayerDefinition], int, int]:
        """Creates layers in Yolov9 model.
        Adapts from original yolov9/models/detect/yolov9.yaml.

        Args:
            in_channels: Number of input channels.
            num_classes: Number of detection classes.

        Returns:
            A tuple of following:
                - List of Layer Definition [(input_layer_idx, layer)]
                - Index of starting auxilary branch in layer def list.
                - Index of starting final detect module in layer def list.
        """
        layers = [
            ## backbone

            # 0. Conv down
            (-1, modules.Conv(in_channels, 64, 3, 2)), # 0-P1/2
            # 1. Conv down
            (-1, modules.Conv(64, 128, 3, 2)), # 1-P2/4
            # 2. Elan-1 block
            (-1, modules.RepNCSPELAN4(128, 256, 128, 64, 1)), # 2
            # 3. Conv down
            (-1, modules.Conv(256, 256, 3, 2)), # 3-P3/8
            # 4. Elan-2 block
            (-1, modules.RepNCSPELAN4(256, 512, 256, 128, 1)),
            # 5. Conv down
            (-1, modules.Conv(512, 512, 3, 2)), # 5-P4/16
            # 6. Elan-2 block
            (-1, modules.RepNCSPELAN4(512, 512, 512, 256, 1)),
            # 7. Conv down
            (-1, modules.Conv(512, 512, 3, 2)), # 7-P5/32
            # 8. Elan-2 block
            (-1, modules.RepNCSPELAN4(512, 512, 512, 256, 1)),

            ## YOLOv9 head

            # 9. Elan-SPP block
            (-1, modules.SPPELAN(512, 512, 256)),
            # 10. Up-concat merge
            (-1, nn.Upsample(2, "nearest")),
            # 11. Concat backbone P4
            ((-1, 6), modules.Concat(-1)),
            # 12. Elan-2 block
            (-1, modules.RepNCSPELAN4(1024, 512, 512, 256, 1)),
            # 13. Up-concat merge
            (-1, nn.Upsample(2, "nearest")),
            # 14. Concat backbone P3
            ((-1, 4), modules.Concat(-1)),
            # 15. Elan-2 block
            (-1, modules.RepNCSPELAN4(1024, 256, 256, 128, 1)), # 15-P3/8-small
            # 16. Conv-down merge
            (-1, modules.Conv(256, 256, 3, 2)),
            # 17. Concat head P4
            ((-1, 12), modules.Concat(-1)),
            # 18. Elan-2 block
            (-1, modules.RepNCSPELAN4(768, 512, 512, 256, 1)), # 18 P4/16-medium
            # 19. Conv-down merge
            ( -1, modules.Conv(512, 512, 3, 2)),
            # 20. Concat haed p5
            ((-1, 9), modules.Concat(-1)),
            # 21. Elan-2 block
            (-1, modules.RepNCSPELAN4(1024, 512, 512, 256, 1)),  # 21 (P5/32-large)

            ## Multi-level reversible auxiliary branch

            # 22. Routing
            (4, modules.CBLinear(512, [256])),
            # 23. Routing
            (6, modules.CBLinear(512, [256, 512])),
            # 24. Routing
            (8, modules.CBLinear(512, [256, 512, 512])),

            # 25. Conv down
            (None, modules.Conv(in_channels, 64, 3, 2 )), # 25-P1/2
            # 26. Conv down
            (-1, modules.Conv(64, 128, 3, 2 )),  # 26-P2/4
            # 27. Elan-1 block
            (-1, modules.RepNCSPELAN4(128, 256, 128, 64, 1)),

            # 28. Conv down
            (-1, modules.Conv(256, 256, 3, 2)), # 28-P3/8
            # 29. CBFuse
            ((22, 23, 24, -1), modules.CBFuse(0, [1.0, 2.0, 4.0])),
            # 30. Elan-2 block
            (-1, modules.RepNCSPELAN4(256, 512, 256, 128, 1)), # A3

            # 31. Conv down fuse
            (-1, modules.Conv(512, 512, 3, 2)), # 31-P4/16
            # 32. Fuse
            ((23, 24, -1), modules.CBFuse(1, [1.0, 2.0])),
            # 33. Elan-2 block
            (-1, modules.RepNCSPELAN4(512, 512, 512, 256, 1)), # A4

            # 34. Conv down
            (-1, modules.Conv(512, 512, 3, 2)), # 34-P5/32
            # 35. CBFuse
            ((24, -1), modules.CBFuse(2, [1.0])),
            # 36. Elan-2 block
            (-1, modules.RepNCSPELAN4(512, 512, 512, 256, 1)), # A5

            ## Detection head

            # 37. Dual Detect (A3, A4, A5, P3, P4, P5)
            (
                (30, 33, 36, 15, 18, 21),
                modules.DualDDetect(num_classes, [512, 512, 512, 256, 512, 512], [8, 16, 32])
            )
        ]

        auxilary_index = 22
        detection_index = len(layers) - 1

        return layers, auxilary_index, detection_index

    @staticmethod
    def create_single_head(num_classes: int) -> LayerDefinition:
        # 37. Single Detect (P3, P4, P5)
        detect_head = (
            (15, 18, -1), modules.DDetect(num_classes, [256, 512, 512], [8, 16, 32])
        )
        return detect_head


class Yolov9CFactory:
    """Factory class to create layers in Yolov9-C model."""
    @staticmethod
    def create_layers(in_channels: int, num_classes: int) -> tuple[list[LayerDefinition], int, int]:
        """Creates layers in Yolov9-C variant model.
        Changes Conv-Down in Plain Yolov9 to Avg-Conv-Down.

        Adapts from original yolov9/models/detect/yolov9-c.yaml.

        Args:
            in_channels: Number of input channels.
            num_classes: Number of detection classes.

        Returns:
            A tuple of following:
                - List of Layer Definition [(input_layer_idx, layer)]
                - Index of starting auxilary branch in layer def list.
                - Index of starting final detect module in layer def list.
        """
        layers = [
            ## YOLOv9 backbone

            # 0. Conv down
            (-1, modules.Conv(3, 64, 3, 2)), # 0-P1/2
            # 1. Conv down
            (-1, modules.Conv(64, 128, 3, 2)), # 1-P2/4
            # 2. Elan-1 block
            (-1, modules.RepNCSPELAN4(128, 256, 128, 64, 1 ) ),
            # 3. Avg-conv down
            (-1, modules.ADown(256, 256)), # 3-P3/8
            # 4. Elan-2 block
            (-1, modules.RepNCSPELAN4(256, 512, 256, 128, 1)),
            # 5. Avg-conv down
            (-1, modules.ADown(512, 512 )), # 5-P4/16
            # 6. Elan-2 block
            (-1, modules.RepNCSPELAN4(512, 512, 512, 256, 1)),
            # 7. Avg-conv down
            (-1, modules.ADown(512, 512)),  # 7-P5/32
            # 8. Elan-2 block
            (-1, modules.RepNCSPELAN4(512, 512, 512, 256, 1)),

            ## YOLOv9 head

            # 9. Elan-spp block
            (-1, modules.SPPELAN(512, 512, 256)),
            # 10. Up-concat merge
            (-1, nn.Upsample(2.0, "nearest")),
            # 11. Concat backbone P4
            ((-1, 6), modules.Concat(-1)),
            # 12. Elan-2 block
            (-1, modules.RepNCSPELAN4(1024, 512, 512, 256, 1)),
            # 13. up-concat merge
            (-1, nn.Upsample(2.0, "nearest")),
            # 14. Concat backbone P3
            ((-1, 4), modules.Concat(-1)),
            # 15. Elan-2 block
            (-1, modules.RepNCSPELAN4(1024, 256, 256, 128, 1)),  # 15 (P3/8-small)
            # 16. avg-conv-down merge
            (-1, modules.ADown(256, 256)),
            # 17. Concat head P4
            ((-1, 12), modules.Concat(-1)),
            # 18. elan-2 block
            (-1, modules.RepNCSPELAN4(768, 512, 512, 256, 1 ) ),  # 18 (P4/16-medium)
            # 19. avg-conv-down merge
            ( -1, modules.ADown(512, 512)),
            # 20. Concat head P5
            ((-1, 9), modules.Concat(-1) ),
            # 21. Elan-2 block
            (-1, modules.RepNCSPELAN4(1024, 512, 512, 256, 1 ) ),  # 21 (P5/32-large)

            ## multi-level reversible auxiliary branch

            # 22. Routinng
            (4, modules.CBLinear(512, [256] )),
            # 23. Routing
            (6, modules.CBLinear(512, [256, 512])),
            # 24. Routing
            (8, modules.CBLinear(512, [256, 512, 512])),

            # 25. Conv down
            (None, modules.Conv(in_channels, 64, 3, 2 )), # 25-P1/2
            # 26. Conv down
            (-1, modules.Conv(64, 128, 3, 2 ) ), # 26-P2/4
            # 27. Elan-1 block
            (-1, modules.RepNCSPELAN4(128, 256, 128, 64, 1)),

            # 28. Avg-conv down fuse
            (-1, modules.ADown(256, 256)), # 28-P3/8
            # 29. CBFuse
            ((22, 23, 24, -1), modules.CBFuse(0, [1.0, 2.0, 4.0])),
            # 30. Elan-2 block
            (-1, modules.RepNCSPELAN4(256, 512, 256, 128, 1)),

            # 31. Avg-conv down fuse
            (-1, modules.ADown(512, 512)), # 31-P4/16
            # 32. CBFuse
            ((23, 24, -1), modules.CBFuse(1, [1.0, 2.0])),
            # 33. Elan-2 block
            (-1, modules.RepNCSPELAN4(512, 512, 512, 256, 1)),

            # 34. Avg-conv down fuse
            (-1, modules.ADown(512, 512)), # 34-P5/32
            # 35. CBFuse
            ((24, -1), modules.CBFuse(2, [1.0])),
            # 36. Elan-2 block
            (-1, modules.RepNCSPELAN4(512, 512, 512, 256, 1)),

            ## Detection head

            # 37. DualDDetect(A3, A4, A5, P3, P4, P5)
            (
                (30, 33, 36, 15, 18, 21),
                modules.DualDDetect(num_classes, [512, 512, 512, 256, 512, 512], [8, 16, 32])
            )
        ]

        auxilary_index = 22
        detection_index = len(layers) - 1

        return layers, auxilary_index, detection_index

    @staticmethod
    def create_single_head(num_classes: int) -> LayerDefinition:
        # 37. Single Detect (P3, P4, P5)
        detect_head = (
            (15, 18, -1), modules.DDetect(num_classes, [256, 512, 512], [8, 16, 32])
        )
        return detect_head


class Yolov9EFactory:
    """Factory class to create layers in Yolov9-E model."""
    @staticmethod
    def create_layers(in_channels: int, num_classes: int) -> tuple[list[LayerDefinition], int, int]:
        """Creates layers in Yolov9-E variant model.
        Deeper than plain Yolov9 and Yolov9-C.

        Adapts from original yolov9/models/detect/yolov9-e.yaml.

        Args:
            in_channels: Number of input channels.
            num_classes: Number of detection classes.

        Returns:
            A tuple of following:
                - List of Layer Definition [(input_layer_idx, layer)]
                - Index of starting auxilary branch in layer def list.
                - Index of ending auxilary branch in layer def list.
        """
        layers = [
            ## YOLOv9 backbone

            # 0. conv down
            (-1, modules.Conv(3, 64, 3, 2)), # 0-P1/2
            # 1. conv down
            (-1, modules.Conv(64, 128, 3, 2)), # 1-P2/4
            # 2. csp-elan block
            (-1, modules.RepNCSPELAN4(128, 256, 128, 64, 2)),
            # 3. avg-conv down
            (-1, modules.ADown(256, 256)),  # 3-P3/8
            # 4. csp-elan block
            (-1, modules.RepNCSPELAN4(256, 512, 256, 128, 2)),
            # 5. avg-conv down
            (-1, modules.ADown(512, 512)),  # 5-P4/16
            # 6. csp-elan block
            (-1, modules.RepNCSPELAN4(512, 1024, 512, 256, 2)),
            # 7. avg-conv down
            (-1, modules.ADown(1024, 1024)),  # 7-P5/32
            # 8. csp-elan block
            (-1, modules.RepNCSPELAN4(1024, 1024, 512, 256, 2)),

            ### routing

            (0, modules.CBLinear(64, [64])), # 9
            (2, modules.CBLinear(256, [64, 128])), # 10
            (4, modules.CBLinear(512, [64, 128, 256])), # 11
            (6, modules.CBLinear(1024, [64, 128, 256, 512])), # 12
            (8, modules.CBLinear(1024, [64, 128, 256, 512, 1024])), # 13

            # 14. Conv down
            (None, modules.Conv(in_channels, 64, 3, 2)), # 14-P1/2
            ((9, 10, 11, 12, 13, -1), modules.CBFuse(0, [1.0, 2.0, 4.0, 8.0, 16.0])), # 15
            # 16. Conv down
            (-1, modules.Conv(64, 128, 3, 2)),  # 16-P2/4
            ((10, 11, 12, 13, -1), modules.CBFuse(1, [1.0, 2.0, 4.0, 8.0])), # 17
            # 18. csp-elan block
            (-1, modules.RepNCSPELAN4(128, 256, 128, 64, 2)),

            # 19. Avg-conv down fuse
            (-1, modules.ADown(256, 256)),  # 19-P3/8
            ((11, 12, 13, -1), modules.CBFuse(2, [1.0, 2.0, 4.0])), # 20
            # 21. csp-elan block
            (-1, modules.RepNCSPELAN4(256, 512, 256, 128, 2)),

            # 22. avg-conv down fuse
            (-1, modules.ADown(512, 512)),  # 22-P4/16
            ((12, 13, -1), modules.CBFuse(3, [1.0, 2.0])), # 23
            # 24. csp-elan block
            (-1, modules.RepNCSPELAN4(512, 1024, 512, 256, 2)),

            # 25. avg-conv down fuse
            (-1, modules.ADown(1024, 1024)), # 25-P5/32
            ((13, -1), modules.CBFuse(4, [1.0])), # 26
            # 27. csp-elan block
            (-1, modules.RepNCSPELAN4(1024, 1024, 512, 256, 2)),

            ## Yolov9 Head

            ### multi-level auxiliary branch

            # 28. elan-spp block
            (8, modules.SPPELAN(1024, 512, 256)),

            # 29. up-concat merge
            (-1, nn.Upsample(2.0, "nearest")),
            ((-1, 6), modules.Concat(-1)), # concat backbone P4

            # 31. csp-elan block
            (-1, modules.RepNCSPELAN4(1536, 512, 512, 256, 2)),

            # 32. up-concat merge
            (-1, nn.Upsample(2.0, "nearest")),
            ((-1, 4), modules.Concat(-1)),  # 33 concat backbone P3

            # 34. csp-elan block
            (-1, modules.RepNCSPELAN4(1024, 256, 256, 128, 2)),

            ### main branch

            # 35. elan-spp block
            (27, modules.SPPELAN(1024, 512, 256)),

            # 36. up-concat merge
            (-1, nn.Upsample(2.0, 'nearest')),
            ((-1, 24), modules.Concat(-1)), # 37 Concat backbone P4

            # 38. csp-elan block
            (-1, modules.RepNCSPELAN4(1536, 512, 512, 256, 2)),

            # 39. up-concat merge
            (-1, nn.Upsample(2.0, 'nearest')),
            ((-1, 21), modules.Concat(-1)), # 40 concat backbone P3

            # 41. csp-elan block
            (-1, modules.RepNCSPELAN4(1024, 256, 256, 128, 2)), # 41 (P3/8-small)

            # 42. avg-conv-down merge
            (-1, modules.ADown(256, 256)),
            ((-1, 38), modules.Concat(-1)), # 43 concat head P4

            # 44. csp-elan block
            (-1, modules.RepNCSPELAN4(768, 512, 512, 256, 2)), # 44 (P4/16-medium)

            # 45. avg-conv-down merge
            (-1, modules.ADown(512, 512)),
            ((-1, 35), modules.Concat(-1)),  # 46 concat head P5

            # 47. csp-elan block
            (-1, modules.RepNCSPELAN4(1024, 512, 1024, 512, 2)),  # 47 (P5/32-large)

            ## Detection head

            # 48. DualDDetect(A3, A4, A5, P3, P4, P5)
            (
                (34, 31, 28, 41, 44, 47),
                modules.DualDDetect(num_classes, [256, 512, 512, 256, 512, 512], [8, 16, 32])
            )
        ]

        auxilary_index = 28
        end_index = 35
        return layers, auxilary_index, end_index

    @staticmethod
    def create_single_head(num_classes: int) -> LayerDefinition:
        # 48. Single Detect (P5)
        detect_head = (
            (41, 44, -1), modules.DDetect(num_classes, [256, 512, 512], [8, 16, 32])
        )
        return detect_head


class YoloBase(nn.Module):
    """Base class for all Yolo Models."""
    def __init__(
        self,
        name: str,
        num_classes: int = 80,
        layers: list[LayerDefinition] = [],
        auxilary_start: int = 0,
        auxilary_end: int = 0,
        detection_head: LayerDefinition | None= None
    ):
        super().__init__()

        self.__name__ = name
        self.num_classes = num_classes
        self._prepare_layers(layers, auxilary_start, auxilary_end, detection_head)

        # Retrieve list of save layer indices
        save_layers = set() # layers needed to be saved in forward process
        for inputs in self.layer_inputs:
            if isinstance(inputs, Iterable):
                for i in inputs:
                    save_layers.add(i)
            elif isinstance(inputs, int) and inputs != -1: # scalar
                save_layers.add(inputs)

        save_layers.remove(-1)
        self.save_layers = save_layers


    def _prepare_layers(
        self,
        layers: list[LayerDefinition],
        auxilary_start: int,
        auxilary_end: int,
        detection_head: LayerDefinition | None
        ) -> None:
        """Creates a list of layers, a list of inputs for each layer.
        And remove Auxilary branch for single model.

        Returns:
            A tuple of
                - A list of layers.
                - A list of input index for each layers.
        """
        layers_, layer_inputs = [], []

        for input, l in layers[:auxilary_start]:
            layer_inputs.append(input)
            layers_.append(l)

        diff = auxilary_end - auxilary_start

        if detection_head is not None:
            # replace detection head
            layers = layers[:-1] + [detection_head]

        for input, l in layers[auxilary_end:]:
            layers_.append(l)
            # change inputs indices
            if isinstance(input, Iterable):
                new_input = tuple(i - diff if i >= auxilary_start else i for i in input)
            elif isinstance(input, int):
                new_input = input - diff if input >= auxilary_start else input
            else:
                new_input = input
            layer_inputs.append(new_input)

        self.layers = layers_
        self.layer_inputs = layer_inputs
        self.num_layers = len(layers_)
        return


    def __call__(self, x):
        ys = {}
        xi = x

        for i, inputs, l in zip(range(self.num_layers), self.layer_inputs, self.layers):
            if inputs is None:
                xi = x
            elif isinstance(inputs, Iterable):
                xi = [ys[j] if j != -1 else xi for j in inputs]
            elif isinstance(inputs, int) and inputs != -1:
                xi = ys[inputs]

            # compute
            xi = l(xi)
            if i in self.save_layers:
                ys[i] = xi

        return xi


class Yolov9(YoloBase):
    """YOLOv9 original model.

    Adapts from original yolov9/models/detect/yolov9.yaml.
    """
    def __init__(self, num_classes: int = 80):
        """Initializes instance of Yolov9 plain model."""
        layers, _ , _ = Yolov9Factory.create_layers(3, num_classes)
        super().__init__("Yolov9", num_classes, layers, len(layers), len(layers))


class Yolov9Converted(YoloBase):
    """YOLOv9 original model which prune auxilary branch.
    """
    def __init__(self, num_classes: int = 80):
        """Initializes instance of Yolov9-converted model."""
        layers, start, end = Yolov9Factory.create_layers(3, num_classes)
        detect = Yolov9Factory.create_single_head(num_classes)
        super().__init__("Yolov9-Converted", num_classes, layers, start, end, detect)


class Yolov9C(YoloBase):
    """YOLOv9-C model.

    Adapts from original yolov9/models/detect/yolov9-c.yaml.
    Changes Conv-Down in Plain Yolov9 to Avg-Conv-Down.
    """
    def __init__(self, num_classes: int = 80):
        """Initializes instance of Yolov9-C model.
        """
        layers, _, _ = Yolov9CFactory.create_layers(3, num_classes)
        super().__init__("Yolov9-C", num_classes, layers, num_classes, num_classes)


class Yolov9CConverted(YoloBase):
    """YOLOv9-C model which prune auxilary branch.
    """
    def __init__(self, num_classes: int = 80):
        """Initializes instance of Yolov9C-converted model."""
        layers, start, end = Yolov9CFactory.create_layers(3, num_classes)
        detect = Yolov9CFactory.create_single_head(num_classes)
        super().__init__("Yolov9-C-Converted", num_classes, layers, start, end, detect)


class Yolov9E(YoloBase):
    """YOLOv9-E model.

    Adapts from original yolov9/models/detect/yolov9-e.yaml.

    Deeper than Yolov9 and Yolov9-C.
    """
    def __init__(self, num_classes: int = 80):
        layers, _, _ = Yolov9EFactory.create_layers(3, num_classes)
        super().__init__("Yolov9-E", num_classes, layers, num_classes, num_classes)


class Yolov9EConverted(YoloBase):
    """YOLOv9-E model which prune auxilary branch.
    """
    def __init__(self, num_classes: int = 80):
        """Initializes instance of Yolov9E-converted model."""
        layers, start, end = Yolov9EFactory.create_layers(3, num_classes)
        detect = Yolov9EFactory.create_single_head(num_classes)
        super().__init__("Yolov9-E-Converted", num_classes, layers, start, end, detect)
