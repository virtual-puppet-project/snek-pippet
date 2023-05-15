# Snek Pippet

Python runner for [mediapipe's](https://github.com/google/mediapipe/tree/master) Face Landmarker
solution.

Data is broadcasted using [PyO3](https://github.com/PyO3/pyo3/tree/main) accelerated
serialization and sending.

## Building

Python 3.9+ and Rust 1.69.0+ are required.

1. Create a venv: `python -m venv venv`
2. Activate the venv:
    * Unix: `source venv/bin/activate`
    * Windows: `source venv/Scripts/activate`
3. Install dependencies: `pip install requirements.txt`
4. Build the PyO3 library:
    * Development: `maturin develop --release`
    * Release: `maturin build --release`
5. Download the mediapipe model file: `curl -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
6. Run the tracker: `python snek-pippet/snek_pippet.py`

