use nalgebra::*;
use pyo3::prelude::*;
use std::iter::FromIterator;

// goals:
// surface/object data structure needs to be designed
// spawn GPU threads for rays

#[pyclass(name=rray)]
struct Ray {
    #[pyo3(get, set)]
    ppower: f64,
    #[pyo3(get, set)]
    ppos: Vec<f64>,
    #[pyo3(get, set)]
    pdir: Vec<f64>,
    #[pyo3(get, set)]
    pcolor: Vec<f64>,
    #[pyo3(get, set)]
    pshadow: bool,

    rpower: f64,
    rpos: Vector3<f64>,
    rdir: Vector3<f64>,
    rcolor: Vector3<f64>,
    rshadow: bool,
}

#[pymethods]
impl Ray {
    #[new]
    fn new() -> Ray {
        Ray {
            ppower: 0.0,
            ppos: vec![0.0; 3],
            pdir: vec![0.0; 3],
            pcolor: vec![0.0; 3],
            pshadow: false,
            rpower: 0.0,
            rpos: Vector3::zeros(),
            rdir: Vector3::zeros(),
            rcolor: Vector3::zeros(),
            rshadow: false,
        }
    }
    fn propagate(&mut self) -> Ray {
        self.update_rray();

        // temp, for testing
        let test_power = 0.1;
        let test_pos = Vector3::new(0.1, 0.1, 0.1);
        let test_dir = Vector3::new(0.1, 0.1, 0.1);
        let test_color = Vector3::new(0.1, 0.2, 0.1);
        let test_shadow = true;

        let result = Ray::new()
            .new_power(test_power)
            .new_pos(test_pos)
            .new_dir(test_dir)
            .new_color(test_color)
            .new_shadow(test_shadow);
        result
    }
}

impl Ray {
    fn update_rray(&mut self) -> () {
        // copies python ray data to rust ray data
        self.rpower = self.ppower;
        self.rpos = Vector3::from_iterator(self.ppos.clone().into_iter());
        self.rdir = Vector3::from_iterator(self.pdir.clone().into_iter());
        self.rcolor = Vector3::from_iterator(self.pcolor.clone().into_iter());
        self.rshadow = self.pshadow;
    }
    fn new_power(mut self, new_power: f64) -> Ray {
        self.ppower = new_power;
        self.rpower = new_power;
        self
    }
    fn new_pos(mut self, new_pos: Vector3<f64>) -> Ray {
        self.ppos = Vec::from_iter(new_pos.into_iter().cloned());
        self.rpos = new_pos;
        self
    }
    fn new_dir(mut self, new_dir: Vector3<f64>) -> Ray {
        self.pdir = Vec::from_iter(new_dir.into_iter().cloned());
        self.rdir = new_dir;
        self
    }
    fn new_color(mut self, new_color: Vector3<f64>) -> Ray {
        self.pcolor = Vec::from_iter(new_color.into_iter().cloned());
        self.rcolor = new_color;
        self
    }
    fn new_shadow(mut self, new_shadow: bool) -> Ray {
        self.pshadow = new_shadow;
        self.rshadow = new_shadow;
        self
    }
}

#[pymodule]
fn rtlib(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Ray>();
    Ok(())
}
