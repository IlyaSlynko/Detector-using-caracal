from caracal import (
    cara_types,
    Event,
    handler,
    Node,
    ProjectInfo,
    Property,
    Session,
)
import logging

from Nodes import Camera, Detector, Vizualizer

if __name__ == "__main__":
    with Session(server_port=2020,) as session:
        # logging.basicConfig(level=logging.DEBUG)
        detetor = Detector()
        cam = Camera()
        visualizer = Vizualizer()

        cam.camera_url = 'rtsp://admin:admin@77.233.1.7:554/cam/realmonitor?channel=1&subtype=0'
        cam.camera_name = "cam_1"
        cam.batch_length = 2
        cam.image_width = 360
        cam.image_height = 360

        detetor.input_image_batch.connect(cam.image_batch)
        visualizer.input_batch.connect(cam.image_batch, detetor.output_rect)

        session.run()