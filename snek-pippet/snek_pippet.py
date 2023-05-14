import numpy
import cv2
import mediapipe as mp
import functools

import pipper

TRACKER_NAME: str = "snek-pippet"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

PORT: int = 8787  # VPUP


def handle_detection(broadcaster: pipper.Broadcaster, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
    if len(result.face_landmarks) < 1 or len(result.face_blendshapes) < 1:
        return

    broadcaster.send(result.face_landmarks[0], result.face_blendshapes[0])


def main() -> None:
    print("Starting {}".format(TRACKER_NAME))

    broadcaster = pipper.Broadcaster(PORT)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit(1)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="models/face_landmarker.task"),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        result_callback=functools.partial(handle_detection, broadcaster),
    )
    detector = FaceLandmarker.create_from_options(
        options
    )

    while True:
        success, frame = cap.read()

        if not success:
            print("Cannot receive frame (stream end?). Exiting.")
            break

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=frame)

        detector.detect_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    cap.release()
    cv2.destroyAllWindows()  # TODO probably not needed


if __name__ == "__main__":
    main()
