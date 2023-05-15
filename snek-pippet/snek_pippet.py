import numpy
import cv2
import mediapipe as mp
import functools
from argparse import ArgumentParser
import json

TRACKER_NAME: str = "snek-pippet"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def handle_detection_rust(broadcaster, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
    if len(result.face_landmarks) < 1 or len(result.face_blendshapes) < 1:
        return

    broadcaster.send(
        result.facial_transformation_matrixes[0].tolist(), result.face_blendshapes[0])


def handle_detection_python(broadcaster, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
    if len(result.facial_transformation_matrixes) < 1 or len(result.face_blendshapes) < 1:
        return

    blendshapes = []
    for i in result.face_blendshapes[0]:
        blendshapes.append({
            "name": i.category_name,
            "score": i.score
        })
    mat4 = result.facial_transformation_matrixes[0].tolist()
    data_str = json.dumps({
        "x": mat4[0],
        "y": mat4[1],
        "z": mat4[2],
        "w": mat4[3],
        "blendshapes": blendshapes
    })

    broadcaster.sendto(str.encode(data_str), ("127.0.0.1", 8787))


def start(camera_index: int, port: int, use_rust: bool) -> None:
    if use_rust:
        import pipper
    else:
        import socket

    print("Starting {}".format(TRACKER_NAME))

    broadcaster = pipper.Broadcaster(port) if use_rust else socket.socket(
        socket.AF_INET, socket.SOCK_DGRAM)

    cap = cv2.VideoCapture(camera_index)
    print("Using backend: {}".format(cap.getBackendName()), flush=True)
    if not cap.isOpened():
        print("Cannot open camera")
        exit(1)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="models/face_landmarker.task"),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        result_callback=functools.partial(
            handle_detection_rust if use_rust else handle_detection_python, broadcaster),
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


if __name__ == "__main__":
    parser = ArgumentParser(description="Runner for mediapipe face landmarker")
    parser.add_argument("--port", "-p", type=int,
                        default=8787, help="The port to broadcast on")
    parser.add_argument("--camera", "-c", type=int,
                        default=0, help="The camera to use")
    parser.add_argument("--list-cameras", action="store_true",
                        help="List the cameras available to use")
    parser.add_argument("--use-python", action="store_true",
                        help="Only use Python to handle broadcasting")

    args = parser.parse_args()
    if args.list_cameras:
        if args.use_python:
            print("Unable to check for cameras in python-only mode")
            exit(1)

        print(json.dumps(pipper.camera.list_cameras(), indent=2))
        exit(0)

    start(camera_index=args.camera, port=args.port, use_rust=not args.use_python)
