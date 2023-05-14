mod camera;

use std::net::{Ipv4Addr, SocketAddrV4, UdpSocket};

use pyo3::prelude::*;
use serde::Serialize;

// TODO will need to figure out head translation/rotation from https://github.com/google/mediapipe/blob/master/mediapipe/tasks/python/vision/face_landmarker.py#L250

#[allow(dead_code)]
#[pyclass]
#[derive(FromPyObject, Serialize)]
struct Category {
    #[serde(skip_serializing)]
    index: u16,
    score: f32,
    #[serde(skip_serializing)]
    display_name: String,
    #[serde(rename = "name")]
    category_name: String,
}

#[allow(dead_code)]
#[pyclass]
#[derive(FromPyObject, Serialize)]
struct Landmark {
    x: f32,
    y: f32,
    z: f32,
    #[serde(skip_serializing)]
    visibility: f32,
    #[serde(skip_serializing)]
    presence: f32,
}

#[derive(Serialize)]
struct WireData {
    landmarks: Vec<Landmark>,
    blendshapes: Vec<Category>,
}

impl From<(Vec<Landmark>, Vec<Category>)> for WireData {
    fn from(value: (Vec<Landmark>, Vec<Category>)) -> Self {
        WireData {
            landmarks: value.0,
            blendshapes: value.1,
        }
    }
}

#[pyclass]
struct Broadcaster {
    socket: UdpSocket,
    target: SocketAddrV4,
}

#[pymethods]
impl Broadcaster {
    #[new]
    fn new(port: u16) -> Self {
        println!("sending to port {}", &port);

        let socket = UdpSocket::bind("0.0.0.0:0").unwrap();
        socket.set_broadcast(true).unwrap();

        Self {
            socket,
            target: SocketAddrV4::new(Ipv4Addr::LOCALHOST, port),
        }
    }

    fn send(&self, landmarks: Vec<Landmark>, blendshapes: Vec<Category>) {
        let _ = self.socket.send_to(
            serde_json::to_string(&WireData::from((landmarks, blendshapes)))
                .unwrap_or_default()
                .as_bytes(),
            self.target,
        );
    }
}

#[pymodule]
fn pipper(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Broadcaster>()?;

    let camera_module = PyModule::new(py, camera::MODULE_NAME)?;
    camera::init(camera_module)?;

    m.add_submodule(camera_module)?;

    Ok(())
}
