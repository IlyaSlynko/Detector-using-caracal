import collections
import pickle

import caracal as cr

from torchvision.models import detection
import numpy as np
import torch
import cv2


class Detector(cr.Node):
    device = cr.Property(cr.cara_types.String(), default_value="cpu")
    confidence = cr.Property(cr.cara_types.Float(), default_value=0.4)

    output_rect = cr.Event("output_rect",
                           (
                               cr.cara_types.List(
                                   cr.cara_types.Tuple(
                                       cr.cara_types.Int(), cr.cara_types.Ndarray()
                                   )
                               ),
                           )
                           )

    @cr.handler(
        "input_image_batch",
        (
                cr.cara_types.List(
                    cr.cara_types.Tuple(cr.cara_types.Int("id_frame"), cr.cara_types.Ndarray("frame"))
                ),
        )
    )
    def input_image_batch(self, msg: cr.Message):
        self.images.append(msg)

    def initialize(self):
        self.images = collections.deque()

        # self.classes = pickle.loads(open(self.classes_path, "rb").read())
        self.colors = np.random.uniform(0, 255, size=(90, 3))
        self.model = detection.retinanet_resnet50_fpn(
            pretrained=True,
            progress=True,
            pretrained_backbone=True,
        ).to(self.device)
        self.model.eval()

    def run(self):

        while True:
            result = []
            if self.images:
                msg = self.images.pop()
                frames = msg.value
                for id_frame, frame in frames:

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.transpose((2, 0, 1))

                    frame = np.expand_dims(frame, axis=0)
                    frame = frame / 255.0
                    frame = torch.FloatTensor(frame)

                    frame = frame.to(self.device)
                    detections = self.model(frame)[0]

                    rect_list = []
                    for i in range(0, len(detections["boxes"])):
                        confidence = detections["scores"][i]

                        if confidence > self.confidence:
                            box = detections["boxes"][i].detach().cpu().numpy()
                            rect_list.append(box)

                    result.append((id_frame, rect_list))

                self.fire(self.output_rect, result, msg_id=msg.id)
