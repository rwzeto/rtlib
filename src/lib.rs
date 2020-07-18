use nalgebra as na;
use pyo3::prelude::*;
use regex as re;
use std;
use std::iter::FromIterator;

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
    rpos: na::Vector3<f64>,
    rdir: na::Vector3<f64>,
    rcolor: na::Vector3<f64>,
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
            rpos: na::Vector3::zeros(),
            rdir: na::Vector3::zeros(),
            rcolor: na::Vector3::zeros(),
            rshadow: false,
        }
    }
    fn propagate(&mut self) -> Ray {
        self.update_rray();

        // temp, for testing
        let test_power = 0.1;
        let test_pos = na::Vector3::new(0.1, 0.1, 0.1);
        let test_dir = na::Vector3::new(0.1, 0.1, 0.1);
        let test_color = na::Vector3::new(0.1, 0.2, 0.1);
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
        self.rpos = na::Vector3::from_iterator(self.ppos.clone().into_iter());
        self.rdir = na::Vector3::from_iterator(self.pdir.clone().into_iter());
        self.rcolor = na::Vector3::from_iterator(self.pcolor.clone().into_iter());
        self.rshadow = self.pshadow;
    }
    fn new_power(mut self, new_power: f64) -> Ray {
        self.ppower = new_power;
        self.rpower = new_power;
        self
    }
    fn new_pos(mut self, new_pos: na::Vector3<f64>) -> Ray {
        self.ppos = Vec::from_iter(new_pos.into_iter().cloned());
        self.rpos = new_pos;
        self
    }
    fn new_dir(mut self, new_dir: na::Vector3<f64>) -> Ray {
        self.pdir = Vec::from_iter(new_dir.into_iter().cloned());
        self.rdir = new_dir;
        self
    }
    fn new_color(mut self, new_color: na::Vector3<f64>) -> Ray {
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

struct RawOBJ {
    vertices: Vec<na::Vector3<f64>>,
    faces: Vec<Vec<u64>>
}

impl RawOBJ {
    fn new() -> RawOBJ {
        RawOBJ { vertices: vec![], faces: vec![] }
    }
}

fn load_obj(filename: &str) -> RawOBJ {
    let document = std::fs::read_to_string(filename).unwrap();
    let vertex_token = re::Regex::new("(v )").unwrap();
    let num_token = re::Regex::new("[+-]?([0-9]*[.])?[0-9]+").unwrap();
    
    let face_token = re::Regex::new("(f )").unwrap();
    let index_token = re::Regex::new("( [0-9]*)").unwrap();

    let mut new_obj: RawOBJ = RawOBJ::new();

    for line in document.lines().into_iter() {

        if vertex_token.is_match(line) {
            let mut coords: na::Vector3<f64> = na::Vector3::zeros();
            for (n, word) in line.split(" ").enumerate() {
                if n > 0 {
                    let cap = num_token.captures(word).unwrap();
                    let text = cap.get(0).map_or("fail", |m| m.as_str());
                    coords[n-1] = text.parse().unwrap();
                }
            }
            new_obj.vertices.push(coords);
        }

        if face_token.is_match(line) {
            let mut vertex_indices: Vec<u64> = vec![];
            for (n, word) in line.split(" ").enumerate() {
                if n > 0 {
                    for (m, intstr) in word.split("/").enumerate() {
                        if m==0 {
                            let vertex_index = intstr.parse::<u64>().unwrap();
                            vertex_indices.push(vertex_index);
                        }
                    }
                }
            }
            new_obj.faces.push(vertex_indices)
        }
    }
    println!("{} vertices with {} faces loaded", new_obj.vertices.len(), new_obj.faces.len());
    new_obj
}

#[pymodule]
fn rtlib(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Ray>();

    #[pyfn(m, "load_obj")]
    fn load_obj_py(_py: Python, filename: &str) -> PyResult<()> {
        let test = load_obj(filename);
        Ok(())
    }

    Ok(())
}
