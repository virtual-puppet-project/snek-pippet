use std::net::{Ipv4Addr, SocketAddrV4, UdpSocket};

use pyo3::prelude::*;
use serde::Serialize;

#[pyclass]
#[derive(FromPyObject, Serialize)]
struct Category {
    index: u16,
    score: f32,
    display_name: String,
    category_name: String,
}

#[pymethods]
impl Category {}

#[pyclass]
#[derive(FromPyObject, Serialize)]
struct Landmark {
    x: f32,
    y: f32,
    z: f32,
    visibility: f32,
    presence: f32,
}

#[pyclass]
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
fn pipper(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Broadcaster>()?;

    Ok(())
}
