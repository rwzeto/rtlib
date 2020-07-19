use nalgebra as na;
use pyo3::prelude::*;
use regex as re;
use std;
use std::iter::FromIterator;

#[pyclass(name=ray)]
struct Ray {
    power: f64,
    pos: na::Vector3<f64>,
    dir: na::Vector3<f64>,
    color: na::Vector3<f64>,
    shadow: bool,
}

#[pymethods]
impl Ray {
    #[new]
    fn new(ppower: f64, ppos: Vec<f64>, pdir: Vec<f64>, pcolor: Vec<f64>, shadow: bool) -> Ray {
        Ray {
            power: 0.0,
            pos: na::Vector3::from_vec(ppos),
            dir: na::Vector3::from_vec(pdir),
            color: na::Vector3::from_vec(pcolor),
            shadow: false
        }
    }
}

struct RawGeometric {
    raw_obj_vertices: Vec<na::Vector3<f64>>,
    raw_obj_faces: Vec<Vec<usize>>
}

impl RawGeometric {
    fn new() -> RawGeometric {
        RawGeometric { raw_obj_vertices: vec![], raw_obj_faces: vec![] }
    }
    fn convert(self) -> Polyhedron {
        let mut polygon_list: Vec<Polygon> = vec![];
        for index_vector in self.raw_obj_faces.into_iter() {
            let M_rows: usize = 3;
            let N_cols: usize = index_vector.len();
            let mut vertices_matrix: na::DMatrix<f64> = na::DMatrix::zeros(M_rows, N_cols);
            for (colnum, index) in index_vector.into_iter().enumerate() {
                let col = self.raw_obj_vertices[index];
                vertices_matrix.set_column(colnum, &col);
            }
            polygon_list.push(Polygon::new(vertices_matrix));
        }
        Polyhedron { faces: polygon_list }
    }
}

#[pyclass(name=poly3d)]
struct Polyhedron {
    faces: Vec<Polygon>
}

#[pymethods]
impl Polyhedron {
    #[new]
    fn new(filename: &str) -> Polyhedron {
        let document = std::fs::read_to_string(filename).unwrap();
        let vertex_token = re::Regex::new("(v )").unwrap();
        let num_token = re::Regex::new("[+-]?([0-9]*[.])?[0-9]+").unwrap();
        
        let face_token = re::Regex::new("(f )").unwrap();
        let index_token = re::Regex::new("( [0-9]*)").unwrap();
    
        let mut new_obj: RawGeometric = RawGeometric::new();
    
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
                new_obj.raw_obj_vertices.push(coords);
            }
    
            if face_token.is_match(line) {
                let mut vertex_indices: Vec<usize> = vec![];
                for (n, word) in line.split(" ").enumerate() {
                    if n > 0 {
                        for (m, intstr) in word.split("/").enumerate() {
                            if m==0 {
                                let vertex_index = intstr.parse::<usize>().unwrap() - 1;
                                vertex_indices.push(vertex_index);
                            }
                        }
                    }
                }
                new_obj.raw_obj_faces.push(vertex_indices)
            }
        }
        println!("{} vertices with {} faces loaded.", new_obj.raw_obj_vertices.len(), new_obj.raw_obj_faces.len());
        new_obj.convert()
    }
}

struct Polygon {
    vertices: na::DMatrix<f64>
}

impl Polygon {
    fn new(vertices_matrix: na::DMatrix<f64>) -> Polygon {
        Polygon { vertices: vertices_matrix }
    }
}



#[pymodule]
fn rtlib(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Ray>();
    m.add_class::<Polyhedron>();

    Ok(())
}
