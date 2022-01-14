import caracal as cr
import cv2


class Camera(cr.Node):
    camera_name = cr.Property(cr.cara_types.String())
    camera_url = cr.Property(cr.cara_types.String())

    image_height = cr.Property(cr.cara_types.Int(), default_value=1920)
    image_width = cr.Property(cr.cara_types.Int(), default_value=1080)

    batch_length = cr.Property(cr.cara_types.Int(), default_value=24)

    image_batch = cr.Event(
        "image_batch",
        (
            cr.cara_types.List(
                cr.cara_types.Tuple(cr.cara_types.Int("id_frame"), cr.cara_types.Ndarray("frame"))
            ),
        )
    )

    def run(self):
        camera = cv2.VideoCapture(self.camera_url)
        id_batch = 0
        while True:
            batch = []
            a = range(1, self.batch_length, 1)
            for id_frame_in_batch in range(1, self.batch_length, 1):
                ret, frame = camera.read()
                if ret != -1:
                    frame = cv2.resize(frame, (self.image_width, self.image_height))
                    batch.append(
                        (id_batch * self.batch_length + id_frame_in_batch, frame)
                    )
            if batch:
                self.fire(self.image_batch, batch)
