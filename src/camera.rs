use nokhwa;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

// TODO unable to figure out how to use nokhwa to pass images to mediapipe without segfaulting

pub const MODULE_NAME: &str = "camera";

#[pyfunction]
fn list_cameras() -> PyResult<Vec<String>> {
    let cameras = match nokhwa::query(nokhwa::native_api_backend().unwrap()) {
        Ok(v) => v,
        Err(e) => return Err(PyRuntimeError::new_err(e.to_string())),
    };

    Ok(cameras
        .into_iter()
        .map(|cam| {
            format!(
                "{} - {}: {}",
                cam.index(),
                cam.human_name(),
                cam.description()
            )
        })
        .collect())
}

pub fn init(m: &PyModule) -> PyResult<()> {
    nokhwa::nokhwa_initialize(|success| {
        if !success {
            println!("Unable to initialize camera helper");
        }
    });
    if !nokhwa::nokhwa_check() {
        return Err(PyRuntimeError::new_err(
            "Unable to initialize camera helper".to_string(),
        ));
    }
    m.add_function(wrap_pyfunction!(list_cameras, m)?)?;

    Ok(())
}
