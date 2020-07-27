use nalgebra as na;
use pyo3::prelude::*;
use regex as re;
use std;
use std::iter::FromIterator;
use std::fmt;
use std::error::Error;
use pyo3::types::{PyList};
use std::cmp::Ordering;

#[pyclass(name=ray)]
#[derive(Clone)]
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
    raw_obj_vertices: Vec<na::RowVector3<f64>>,
    raw_obj_faces: Vec<Vec<usize>>
}

impl RawGeometric {
    fn new() -> RawGeometric {
        RawGeometric { raw_obj_vertices: vec![], raw_obj_faces: vec![] }
    }
    fn convert(self) -> Polyhedron {
        let position: na::Vector3<f64> = na::Vector3::new(0.0, 0.0, 0.0);
        let mut polyhedron_face_list: Vec<Polygon> = vec![];
        let mut polyhedron_vertices_matrix: na::DMatrix<f64> = na::DMatrix::zeros(self.raw_obj_vertices.len(), 3);
        let mut polyhedron_indices_matrix: na::DMatrix<usize> = na::DMatrix::zeros(self.raw_obj_faces.len(), 3);
        for (index_num, index_vector) in self.raw_obj_faces.into_iter().enumerate() {
            let rv: na::RowVector3<usize> = na::RowVector3::from_vec(index_vector.clone());
            polyhedron_indices_matrix.set_row(index_num, &rv);
            let M_rows: usize = index_vector.len();
            let N_cols: usize = 3;
            let mut polygon_vertices_matrix: na::DMatrix<f64> = na::DMatrix::zeros(M_rows, N_cols);
            for (rownum, index) in index_vector.into_iter().enumerate() {
                let row = self.raw_obj_vertices[index];
                polygon_vertices_matrix.set_row(rownum, &row);
            }
            polyhedron_face_list.push(Polygon::new(polygon_vertices_matrix));
        }
        for (vertex_num, vertex_vector) in self.raw_obj_vertices.into_iter().enumerate() {
            polyhedron_vertices_matrix.set_row(vertex_num, &vertex_vector);
        }
        Polyhedron { faces: polyhedron_face_list, vertices: polyhedron_vertices_matrix, indices: polyhedron_indices_matrix, position: position }
    }
}

#[pyclass(name=scene)]
struct Scene {
    objects: Vec<Polyhedron>
}

#[pymethods]
impl Scene {
    #[new]
    fn new() -> Scene {
        Scene { objects: vec![] }
    }
    fn add(&mut self, polyhedron: Polyhedron) -> PyResult<()> {
        self.objects.push(polyhedron);
        Ok(())
    }
    fn propagate(&self, ray: &Ray) -> PyResult<()> {
        // todo: nohit handling
        let hit_face: &Polygon = self.calculate_hit(ray).unwrap();
        println!("{}", hit_face.vertices);
        println!("{}", hit_face.normal);
        Ok(())
    }
}

impl Scene {
    fn calculate_hit(&self, ray: &Ray) -> Result<&Polygon, NoHit> {
        // todo: add filtering 
        for polyhedron in self.objects.iter() {
            let faces = &polyhedron.faces;
            let mut intersections: Vec<na::Vector3<f64>> = vec![];
            let mut matching_faces: Vec<&Polygon> = vec![];
            for face in faces.iter() {
                let point_of_intersection = face.intersection(&ray);
                if face.is_inside(&point_of_intersection) {
                    if (point_of_intersection - ray.pos).dot(&ray.dir) > 0.0 {
                        matching_faces.push(&face);
                    }
                }
            }
            let mut result: Vec<f64> = matching_faces.clone().into_iter().map(|x| (*x).distance_from_ray(ray)).collect();
            let index_of_min: Option<usize> = result
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(index, _)| index);
            let hit_face_index: usize = index_of_min.unwrap();
            let return_face = matching_faces[hit_face_index];
            return Ok(return_face);
        }
        Err(NoHit::new("No hit."))
    }
    // fn calculate_next_rays
}

#[derive(Debug)]
struct NoHit {
    details: String
}

impl NoHit {
    fn new(msg: &str) -> NoHit {
        NoHit { details: msg.to_string() }
    }
}

impl fmt::Display for NoHit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,"{}",self.details)
    }
}

impl Error for NoHit {
    fn description(&self) -> &str {
        &self.details
    }
}

#[pyclass(name=poly3d)]
#[derive(Clone)]
struct Polyhedron {
    faces: Vec<Polygon>,
    vertices: na::DMatrix<f64>,
    indices: na::DMatrix<usize>,
    position: na::Vector3<f64>
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
                let mut coords: na::RowVector3<f64> = na::RowVector3::zeros();
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

fn normalize(vector: na::Vector3<f64>) -> na::Vector3<f64> {
    let norm: f64 = (vector[0].powf(2.0) + vector[1].powf(2.0) + vector[2].powf(2.0)).sqrt();
    assert![norm > 0.0];
    vector / norm
}

fn normal_cross(v1: na::Vector3<f64>, v2: na::Vector3<f64>) -> na::Vector3<f64> {
    let mut cross_product = v1.cross(&v2);
    let mut normal: na::Vector3<f64> = na::Vector3::zeros();
    normal.copy_from(&cross_product);
    normal = normalize(normal);
    normal
}

#[derive(Clone)]
struct Polygon {
    vertices: na::DMatrix<f64>,
    normal: na::Vector3<f64>
}

impl Polygon {
    fn new(vertices_matrix: na::DMatrix<f64>) -> Polygon {
        let edge_0 = vertices_matrix.row(1) - vertices_matrix.row(0);
        let edge_1 = vertices_matrix.row(2) - vertices_matrix.row(1);
        let mut cross_product = edge_0.transpose().cross(&edge_1.transpose());
        let mut normal: na::Vector3<f64> = na::Vector3::zeros();
        normal.copy_from(&cross_product);
        normal = normalize(normal);
        Polygon { 
            vertices: vertices_matrix,
            normal: normal,
        }
    }
    fn intersection(&self, ray: &Ray) -> na::Vector3<f64> {
        let point_on_plane = self.get_vertex(0);
        // minus sign issue here
        let t: f64 = (self.normal.dot(&point_on_plane) - self.normal.dot(&ray.pos))/self.normal.dot(&ray.dir);
        ray.pos+ray.dir*t
    }
    fn get_vertex(&self, vertex_index: usize) -> na::Vector3<f64> {
        let point = self.vertices.row(vertex_index).transpose();
        let mut return_vec: na::Vector3<f64> = na::Vector3::zeros();
        return_vec.copy_from(&point);
        return_vec
    }
    fn is_inside(&self, point_of_intersection: &na::Vector3<f64>) -> bool {
        // todo: add bounds check here
        let mut edges: Vec<na::Vector3<f64>> = vec![];
        edges.push(normalize(self.get_vertex(1)-self.get_vertex(0)));
        edges.push(normalize(self.get_vertex(2)-self.get_vertex(1)));
        edges.push(normalize(self.get_vertex(0)-self.get_vertex(2)));
        let plane_normal: na::Vector3<f64> = normal_cross(edges[0], edges[1]);
        let mut results: Vec<f64> = vec![0.0, 0.0, 0.0];
        for (edge_num, edge) in edges.clone().into_iter().enumerate() {
            let edge_normal: na::Vector3<f64> = normal_cross(edge, plane_normal);
            results[edge_num] = (self.get_vertex(edge_num) - point_of_intersection).dot(&edge_normal);
        }
        let mut is_inside_on_all_sides = false;
        let mut inside_flag: Vec<bool> = vec![false, false, false];
        let mut edge_clip_flag = false;
        for (edgenum, result) in results.clone().into_iter().enumerate() {
            if result.is_nan() {
                inside_flag[edgenum] = false;
            }
            else if result > 0.0 {
                inside_flag[edgenum] = true;
            }
            if result == 0.0 {
                edge_clip_flag = true;
            }
        }
        is_inside_on_all_sides = inside_flag.into_iter().all(|x| x);
        if is_inside_on_all_sides && edge_clip_flag {
            assert![1==0]
        }
        is_inside_on_all_sides
    }
    fn distance_from_ray(&self, ray: &Ray) -> f64 {
        let intersection = self.intersection(ray);
        let distance = (ray.pos - intersection).norm();
        distance
    }
}

#[pymodule]
fn rtlib(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Ray>();
    m.add_class::<Polyhedron>();
    m.add_class::<Scene>();

    Ok(())
}
