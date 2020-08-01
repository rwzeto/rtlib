use nalgebra as na;
use pyo3::prelude::*;
use regex as re;
use std;
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

// convenience function for converting pylist to matrix
fn list_to_matrix(input: Vec<Vec<f64>>) -> na::Matrix3<f64> {
    let mut return_matrix: na::Matrix3<f64> = na::Matrix3::zeros();
    for (rownum, row) in input.iter().enumerate() {
        let rowvec: na::RowVector3<f64> = na::RowVector3::from_vec(input[rownum].clone());
        return_matrix.set_row(rownum, &rowvec);
    }
    return_matrix
}

// convenience function for normalizing a vector
fn normalize(vector: na::Vector3<f64>) -> na::Vector3<f64> {
    let norm: f64 = (vector[0].powf(2.0) + vector[1].powf(2.0) + vector[2].powf(2.0)).sqrt();
    assert![norm > 0.0];
    vector / norm
}

// convenience function for a normalized cross product
fn normal_cross(v1: na::Vector3<f64>, v2: na::Vector3<f64>) -> na::Vector3<f64> {
    let cross_product = v1.cross(&v2);
    let mut normal: na::Vector3<f64> = na::Vector3::zeros();
    normal.copy_from(&cross_product);
    normal = normalize(normal);
    normal
}

fn reflection_matrix(nhat: na::Vector3<f64>) -> na::Matrix3<f64> {
    na::Matrix3::identity() - 2.0 * nhat * nhat.transpose()
}

#[pyclass(name=ray)]
#[derive(Clone, Debug)]
struct Ray {
    power: f64,
    index: f64,
    pos: na::Vector3<f64>,
    dir: na::Vector3<f64>,
    color: na::Vector3<f64>,
    shadow: bool,
}

#[pymethods]
impl Ray {
    #[new]
    fn new(
        ppower: f64,
        pindex: f64,
        ppos: Vec<f64>,
        pdir: Vec<f64>,
        pcolor: Vec<f64>,
        shadow: bool,
    ) -> Ray {
        Ray {
            power: ppower,
            index: pindex,
            pos: na::Vector3::from_vec(ppos),
            dir: na::Vector3::from_vec(pdir),
            color: na::Vector3::from_vec(pcolor),
            shadow: false,
        }
    }
}

impl fmt::Display for Ray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "-- ray pow: {}\npos: [{}, {}, {}]\ndir: [{}, {}, {}]",
            self.power,
            self.pos[0],
            self.pos[1],
            self.pos[2],
            self.dir[0],
            self.dir[1],
            self.dir[2]
        )
    }
}

// convenience struct for importing geometry. only used in Polyhedron::new()
struct RawGeometric {
    raw_obj_vertices: Vec<na::RowVector3<f64>>,
    raw_obj_faces: Vec<Vec<usize>>,
}

impl RawGeometric {
    fn new() -> RawGeometric {
        RawGeometric {
            raw_obj_vertices: vec![],
            raw_obj_faces: vec![],
        }
    }

    // conversion from RawGeometric to Polyhedron
    fn convert(self, index: f64, pos: Vec<f64>, rot: Vec<Vec<f64>>) -> Polyhedron {
        let mut polyhedron_face_list: Vec<Polygon> = vec![];
        let rot_matrix: na::Matrix3<f64> = list_to_matrix(rot);
        for (index_num, index_vector) in self.raw_obj_faces.into_iter().enumerate() {
            let M_rows: usize = index_vector.len();
            let N_cols: usize = 3;
            let mut polygon_vertices_matrix: na::DMatrix<f64> = na::DMatrix::zeros(M_rows, N_cols);
            for (rownum, index) in index_vector.into_iter().enumerate() {
                let mut row =
                    self.raw_obj_vertices[index] + na::RowVector3::<f64>::from_vec(pos.clone());
                let mut tmp = row.transpose();
                tmp = rot_matrix * tmp;
                row = tmp.transpose();
                polygon_vertices_matrix.set_row(rownum, &row);
            }
            polyhedron_face_list.push(Polygon::new(polygon_vertices_matrix));
        }
        Polyhedron {
            id: 0,
            index: index,
            rot: rot_matrix,
            pos: na::Vector3::from_vec(pos),
            faces: polyhedron_face_list,
        }
    }
}

#[pyclass(name=raytree)]
struct RayTree {
    ray: Ray,
    childs: Vec<RayTree>,
}

impl RayTree {
    fn new(first_ray_segment: Ray) -> RayTree {
        RayTree {
            ray: first_ray_segment,
            childs: vec![],
        }
    }
    fn push_child(&mut self, child_node: RayTree) -> () {
        self.childs.push(child_node);
    }
}

#[pymethods]
impl RayTree {
    fn print(&self) -> () {
        if self.childs.len() == 0 {
            return ();
        }
        println!("{}", self.ray);
        for child in self.childs.iter() {
            child.print();
        }
    }
}

// python scene object. polyhedrons are added to it using scene.add() on the python side
#[pyclass(name=scene)]
struct Scene {
    polyhedrons: Vec<Polyhedron>,
}

#[pymethods]
impl Scene {
    #[new]
    fn new() -> Scene {
        Scene {
            polyhedrons: vec![],
        }
    }

    fn add(&mut self, mut polyhedron: Polyhedron) -> PyResult<()> {
        let polyhedron_id: usize = self.polyhedrons.len();
        for polygon in polyhedron.faces.iter_mut() {
            polygon.parent = polyhedron_id;
        }
        self.polyhedrons.push(polyhedron);
        Ok(())
    }

    // main loop for ray tracing in a scene
    fn propagate(&self, ray: Ray) -> RayTree {
        let hit_face: &Polygon = match self.calculate_hit(&ray) {
            Ok(hit_face) => hit_face,
            Err(nohit) => {
                return RayTree {
                    ray: ray,
                    childs: vec![],
                }
            }
        };
        let n = self.polyhedrons[hit_face.parent].index;
        let next_rays: Vec<Ray> = hit_face.calculate_next_rays(&ray, n);
        let mut result = RayTree {
            ray: ray,
            childs: vec![],
        };
        for next_ray in next_rays.into_iter() {
            result.push_child(self.propagate(next_ray));
        }
        result
    }
}

impl Scene {
    fn calculate_hit(&self, ray: &Ray) -> Result<&Polygon, NoHit> {
        // todo: add filtering
        for polyhedron in self.polyhedrons.iter() {
            let faces = &polyhedron.faces;
            let intersections: Vec<na::Vector3<f64>> = vec![];
            let mut matching_faces: Vec<&Polygon> = vec![];
            for face in faces.iter() {
                let point_of_intersection = face.intersection(&ray);
                if face.is_inside(&point_of_intersection) {
                    if (point_of_intersection - ray.pos).dot(&ray.dir) > 0.0 {
                        matching_faces.push(&face);
                    }
                }
            }
            if matching_faces.len() > 0 {
                let result: Vec<f64> = matching_faces
                    .clone()
                    .into_iter()
                    .map(|x| (*x).distance_from_ray(ray))
                    .collect();
                let index_of_min: Option<usize> = result
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(index, _)| index);
                let hit_face_index: usize = index_of_min.unwrap();
                let return_face = matching_faces[hit_face_index];
                return Ok(return_face);
            }
        }
        Err(NoHit::new("No hit."))
    }
    // todo: fn calculate_next_rays
}

// 3d object class for python
#[pyclass(name=poly3d)]
#[derive(Clone)]
struct Polyhedron {
    index: f64,
    id: usize,
    faces: Vec<Polygon>,
    pos: na::Vector3<f64>,
    rot: na::Matrix3<f64>,
}

#[pymethods]
impl Polyhedron {
    // constructor parses .obj file into RawGeometry struct,
    // and then converts that into Polyhedron and Polygon structs
    #[new]
    fn new(filename: &str, index: f64, pos: Vec<f64>, rot: Vec<Vec<f64>>) -> Polyhedron {
        let document = std::fs::read_to_string(filename).unwrap();
        let vertex_token = re::Regex::new("(v )").unwrap();
        let num_token = re::Regex::new("[+-]?([0-9]*[.])?[0-9]+").unwrap();

        let face_token = re::Regex::new("(f )").unwrap();

        let mut new_obj: RawGeometric = RawGeometric::new();

        for line in document.lines().into_iter() {
            if vertex_token.is_match(line) {
                let mut coords: na::RowVector3<f64> = na::RowVector3::zeros();
                for (n, word) in line.split(" ").enumerate() {
                    if n > 0 {
                        let cap = num_token.captures(word).unwrap();
                        let text = cap.get(0).map_or("fail", |m| m.as_str());
                        coords[n - 1] = text.parse().unwrap();
                    }
                }
                new_obj.raw_obj_vertices.push(coords);
            }

            if face_token.is_match(line) {
                let mut vertex_indices: Vec<usize> = vec![];
                for (n, word) in line.split(" ").enumerate() {
                    if n > 0 {
                        for (m, intstr) in word.split("/").enumerate() {
                            if m == 0 {
                                let vertex_index = intstr.parse::<usize>().unwrap() - 1;
                                vertex_indices.push(vertex_index);
                            }
                        }
                    }
                }
                new_obj.raw_obj_faces.push(vertex_indices)
            }
        }
        new_obj.convert(index, pos, rot)
    }
}

// struct representing a collection of three vertices
#[derive(Clone)]
struct Polygon {
    vertices: na::DMatrix<f64>,
    normal: na::Vector3<f64>,
    parent: usize,
}

impl Polygon {
    // this is called when the polyhedron is generating its facets
    fn new(vertices_matrix: na::DMatrix<f64>) -> Polygon {
        let edge_0 = vertices_matrix.row(1) - vertices_matrix.row(0);
        let edge_1 = vertices_matrix.row(2) - vertices_matrix.row(1);
        let cross_product = edge_0.transpose().cross(&edge_1.transpose());
        let mut normal: na::Vector3<f64> = na::Vector3::zeros();
        normal.copy_from(&cross_product);
        normal = normalize(normal);
        Polygon {
            vertices: vertices_matrix,
            normal: normal,
            parent: 0,
        }
    }
    // intersection of ray with self
    fn intersection(&self, ray: &Ray) -> na::Vector3<f64> {
        let point_on_plane = self.get_vertex(0);
        // minus sign issue here
        let t: f64 = (self.normal.dot(&point_on_plane) - self.normal.dot(&ray.pos))
            / self.normal.dot(&ray.dir);
        ray.pos + ray.dir * t
    }
    // convenience function for pulling a specific vertex
    fn get_vertex(&self, vertex_index: usize) -> na::Vector3<f64> {
        let point = self.vertices.row(vertex_index).transpose();
        let mut return_vec: na::Vector3<f64> = na::Vector3::zeros();
        return_vec.copy_from(&point);
        return_vec
    }
    // hit detection between ray and self
    fn is_inside(&self, point_of_intersection: &na::Vector3<f64>) -> bool {
        // todo: add bounds check here
        let mut edges: Vec<na::Vector3<f64>> = vec![];
        edges.push(normalize(self.get_vertex(1) - self.get_vertex(0)));
        edges.push(normalize(self.get_vertex(2) - self.get_vertex(1)));
        edges.push(normalize(self.get_vertex(0) - self.get_vertex(2)));
        let plane_normal: na::Vector3<f64> = normal_cross(edges[0], edges[1]);
        let mut results: Vec<f64> = vec![0.0, 0.0, 0.0];
        for (edge_num, edge) in edges.clone().into_iter().enumerate() {
            // edge is the tangent vector
            // edge_normal is the outward-oriented normal vector in the plane of the polygon
            let edge_normal: na::Vector3<f64> = normal_cross(edge, plane_normal);
            results[edge_num] =
                (self.get_vertex(edge_num) - point_of_intersection).dot(&edge_normal);
        }
        // POI is inside if the dot product between the edge normals and a vector pointing from POI to the edge
        // is positive for each side
        // if the dot product is ever zero, then the ray is clipping an edge
        let mut is_inside_on_all_sides = false;
        let mut inside_flag: Vec<bool> = vec![false, false, false];
        let mut edge_clip_flag = false;
        for (edgenum, result) in results.clone().into_iter().enumerate() {
            if result.is_nan() {
                inside_flag[edgenum] = false;
            } else if result > 0.0 {
                inside_flag[edgenum] = true;
            }
            if result == 0.0 {
                edge_clip_flag = true;
            }
        }
        is_inside_on_all_sides = inside_flag.into_iter().all(|x| x);
        if is_inside_on_all_sides && edge_clip_flag {
            // need to handle this later, but for now, just panic if we clip an edge
            assert![1 == 0]
        }
        is_inside_on_all_sides
    }
    fn distance_from_ray(&self, ray: &Ray) -> f64 {
        let intersection = self.intersection(ray);
        let distance = (ray.pos - intersection).norm();
        distance
    }
    fn calculate_next_rays(&self, incident_ray: &Ray, n: f64) -> Vec<Ray> {
        // reflected ray
        let mut exiting: bool = false;
        let mut n_relative: f64 = 1.0 / n;
        let mut surface_normal: na::Vector3<f64> = self.normal;
        if incident_ray.dir.dot(&self.normal) > 0.0 {
            exiting = true;
            n_relative = n;
        }
        let mut reflected_ray = Ray {
            power: incident_ray.power,
            index: 0.0,
            pos: self.intersection(incident_ray),
            dir: incident_ray.dir,
            color: incident_ray.color,
            shadow: false,
        };
        let mut refracted_ray = Ray {
            power: incident_ray.power * 0.1,
            index: 0.0,
            pos: self.intersection(incident_ray),
            dir: incident_ray.dir,
            color: incident_ray.color,
            shadow: false,
        };
        let reflected_dir = reflection_matrix(surface_normal) * incident_ray.dir;
        let refracted_dir = ( -n_relative * surface_normal.dot(&incident_ray.dir)
            - (1.0 - n_relative.powf(2.0) * (1.0 - surface_normal.dot(&incident_ray.dir).powf(2.0)))
            .sqrt())
            * surface_normal
            + n_relative * incident_ray.dir;
        
        vec![reflected_ray, refracted_ray]
    }
}

// error class incase no surface is hit
#[derive(Debug)]
struct NoHit {
    details: String,
}

impl NoHit {
    fn new(msg: &str) -> NoHit {
        NoHit {
            details: msg.to_string(),
        }
    }
}

impl fmt::Display for NoHit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for NoHit {
    fn description(&self) -> &str {
        &self.details
    }
}

#[pymodule]
fn rtlib(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Ray>();
    m.add_class::<Polyhedron>();
    m.add_class::<Scene>();
    m.add_class::<RayTree>();
    Ok(())
}
