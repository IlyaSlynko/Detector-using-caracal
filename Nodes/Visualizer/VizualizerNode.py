import caracal as cr
import numpy as np
import cv2


class Vizualizer(cr.Node):
    @cr.handler("input_batch",  (
            cr.cara_types.List(
                cr.cara_types.Tuple(cr.cara_types.Int("id_frame"), cr.cara_types.Ndarray("frame"))
            ),
        ), receives_multiple=True, info=None)
    def input_batch(self, msgs):
        for frame, rectangles in zip(msgs.value[0].value, msgs.value[1].value):
            for rect in rectangles[1]:
                if len(rect):
                    (startX, startY, endX, endY) = rect.astype("int")
                    cv2.rectangle(frame[1], (startX, startY), (endX, endY), 10, 2)
            cv2.imshow("stream", frame[1])
            cv2.waitKey(33)

