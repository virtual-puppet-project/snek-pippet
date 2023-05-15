mod camera;

use std::net::{Ipv4Addr, SocketAddrV4, UdpSocket};

use pyo3::prelude::*;
use serde::{ser::SerializeSeq, Serialize};

// TODO will need to figure out head translation/rotation from https://github.com/google/mediapipe/blob/master/mediapipe/tasks/python/vision/face_landmarker.py#L250

#[allow(dead_code)]
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

#[derive(FromPyObject)]
struct PyTransformationMatrix(Vec<Vec<f32>>);

struct Vector4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl Serialize for Vector4 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut s = serializer.serialize_seq(Some(4))?;
        s.serialize_element(&self.x)?;
        s.serialize_element(&self.y)?;
        s.serialize_element(&self.z)?;
        s.serialize_element(&self.w)?;

        s.end()
    }
}

#[derive(Serialize)]
struct TransformationMatrix {
    x: Vector4,
    y: Vector4,
    z: Vector4,
    w: Vector4,
}

impl From<PyTransformationMatrix> for TransformationMatrix {
    fn from(value: PyTransformationMatrix) -> Self {
        let value = value.0;
        Self {
            x: Vector4 {
                x: value[0][0],
                y: value[0][1],
                z: value[0][2],
                w: value[0][3],
            },
            y: Vector4 {
                x: value[1][0],
                y: value[1][1],
                z: value[1][2],
                w: value[1][3],
            },
            z: Vector4 {
                x: value[2][0],
                y: value[2][1],
                z: value[2][2],
                w: value[2][3],
            },
            w: Vector4 {
                x: value[3][0],
                y: value[3][1],
                z: value[3][2],
                w: value[3][3],
            },
        }
    }
}

#[derive(Serialize)]
struct WireData {
    #[serde(flatten)]
    transformation_matrix: TransformationMatrix,
    blendshapes: Vec<Category>,
}

impl From<(PyTransformationMatrix, Vec<Category>)> for WireData {
    fn from(value: (PyTransformationMatrix, Vec<Category>)) -> Self {
        WireData {
            transformation_matrix: TransformationMatrix::from(value.0),
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

    fn send(&self, transformation_matrix: PyTransformationMatrix, blendshapes: Vec<Category>) {
        let _ = self.socket.send_to(
            serde_json::to_string(&WireData::from((transformation_matrix, blendshapes)))
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
