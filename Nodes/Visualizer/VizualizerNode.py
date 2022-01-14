import caracal as cr
import cv2


class Vizualizer(cr.Node):
    @cr.handler("input_batch",  (
            cr.cara_types.List(
                cr.cara_types.Tuple(cr.cara_types.Int("id_frame"), cr.cara_types.Ndarray("frame"))
            ),
        ), receives_multiple=False, info=None)
    def input_batch(self, msg):
        for _, frame in msg.value:
            cv2.imshow("stream", frame)
            cv2.waitKey(33)

