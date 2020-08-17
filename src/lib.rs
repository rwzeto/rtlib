use nalgebra as na;
use pyo3::prelude::*;
use regex as re;
use std;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;
use std::rc::Rc;

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

fn distance(p1: na::Vector3<f64>, p2: na::Vector3<f64>) -> f64 {
    let diff = p2 - p1;
    (diff[0].powf(2.0) + diff[1].powf(2.0) + diff[2].powf(2.0)).sqrt()
}

#[pyclass(name=ray)]
#[derive(Clone, Debug)]
struct Ray {
    power: f64,
    index: f64,
    pos: na::Vector3<f64>,
    dir: na::Vector3<f64>,
    color: na::Vector3<f64>,
    last_ray: bool,
    first_ray: bool,
}

#[pymethods]
impl Ray {
    #[new]
    fn new(index: f64, pos: Vec<f64>, dir: Vec<f64>, color: Vec<f64>) -> Ray {
        Ray {
            power: 1.0,
            index: index,
            pos: na::Vector3::from_vec(pos),
            dir: na::Vector3::from_vec(dir),
            color: na::Vector3::from_vec(color),
            last_ray: false,
            first_ray: true,
        }
    }
}

impl Ray {
    fn shadow(pos: na::Vector3<f64>, dir: na::Vector3<f64>) -> Ray {
        Ray {
            power: 1.0,
            index: 0.0,
            pos: pos,
            dir: normalize(dir),
            color: na::Vector3::<f64>::zeros(),
            first_ray: true,
            last_ray: false,
        }
    }
}

impl fmt::Display for Ray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "-- ray pow: {}, n: {}, 1st: {}\npos: [{}, {}, {}]\ndir: [{}, {}, {}]",
            self.power,
            self.index,
            self.first_ray,
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
    fn convert(
        self,
        index: f64,
        pos: Vec<f64>,
        rot: Vec<Vec<f64>>,
        transmissivity: f64,
        reflectivity: f64,
    ) -> Polyhedron {
        let mut polyhedron_face_list: Vec<Polygon> = vec![];
        let rot_matrix: na::Matrix3<f64> = list_to_matrix(rot);
        for index_vector in self.raw_obj_faces.into_iter() {
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
            transmissivity: transmissivity,
            reflectivity: reflectivity,
        }
    }
}

// python scene object. polyhedrons are added to it using scene.add() on the python side
#[pyclass(name=scene)]
struct Scene {
    polyhedrons: Vec<Polyhedron>,
    lights: Vec<Light>,
}

#[pymethods]
impl Scene {
    #[new]
    fn new() -> Scene {
        Scene {
            polyhedrons: vec![],
            lights: vec![],
        }
    }

    fn add_obj(&mut self, mut polyhedron: Polyhedron) -> PyResult<()> {
        let polyhedron_id: usize = self.polyhedrons.len();
        for polygon in polyhedron.faces.iter_mut() {
            polygon.parent = polyhedron_id;
        }
        self.polyhedrons.push(polyhedron);
        Ok(())
    }
    fn add_light(&mut self, light: Light) -> PyResult<()> {
        self.lights.push(light);
        Ok(())
    }
    fn render(&self, ray: &Ray) -> PyResult<f64> {
        let storage_pixel: Rc<RefCell<f64>> = Rc::new(RefCell::new(0.0));
        Ok(self.propagate(ray, &self.lights, Rc::clone(&storage_pixel)))
    }
}

impl Scene {
    // main loop for ray tracing in a scene
    fn propagate(&self, ray: &Ray, lights: &Vec<Light>, pixel: Rc<RefCell<f64>>) -> f64 {
        if ray.last_ray == false {
            let hit_face: &Polygon = match self.calculate_hit(&ray) {
                Ok(hit_face) => hit_face,
                Err(_) => {
                    return self.last_ray_calc(ray, lights, Rc::clone(&pixel));
                }
            };
            let n = self.polyhedrons[hit_face.parent].index;
            let transmissivity = self.polyhedrons[hit_face.parent].transmissivity;
            let reflectivity = self.polyhedrons[hit_face.parent].reflectivity;
            let next_rays = hit_face.calculate_next_rays(&ray, n, transmissivity, reflectivity);
            for nextray in next_rays.iter() {
                return self.propagate(nextray, lights, Rc::clone(&pixel));
            }
        } else {
            return self.last_ray_calc(ray, lights, Rc::clone(&pixel));
        }
        assert![1 == 0, "propagate logic issue"];
        0.0
    }
    fn last_ray_calc(&self, ray: &Ray, lights: &Vec<Light>, pixel: Rc<RefCell<f64>>) -> f64 {
        let mut storage_pixel = *pixel.borrow_mut();
        let mut pixel_adjustment: f64 = 0.0;
        if !ray.first_ray {
            for light in lights.iter() {
                let shadow_ray: Ray = Ray::shadow(ray.pos, light.pos - ray.pos);
                let obstructions = self.sample_shadow_ray_path(&shadow_ray);
                let dist: f64 = distance(ray.pos, light.pos);
                pixel_adjustment += obstructions * light.power / dist.powf(2.0) * ray.power;
            }
        }
        storage_pixel = storage_pixel + pixel_adjustment;
        storage_pixel
    }
    // for now, keep it simple.
    // sample surface properties in a straight line to the light source
    fn sample_shadow_ray_path(&self, shadow_ray: &Ray) -> f64 {
        // from ray.pos to light source
        // calculate intersections
        let mut transmission_coefficient: f64 = 1.0;
        let hit_faces: Vec<&Polygon> = match self.calculate_all_forward_hits(shadow_ray) {
            Ok(hit_face) => hit_face,
            Err(_) => return 1.0,
        };
        for face in hit_faces.into_iter() {
            let transmissivity = self.polyhedrons[face.parent].transmissivity;
            transmission_coefficient = transmission_coefficient * transmissivity;
        }
        transmission_coefficient
    }
    fn calculate_all_forward_hits(&self, ray: &Ray) -> Result<Vec<&Polygon>, NoHit> {
        let mut matching_faces: Vec<&Polygon> = vec![];
        for polyhedron in self.polyhedrons.iter() {
            let faces = &polyhedron.faces;
            for face in faces.iter() {
                let point_of_intersection = face.intersection(&ray);
                if face.is_inside(&point_of_intersection) {
                    if (point_of_intersection - ray.pos).dot(&ray.dir) > 0.0 {
                        matching_faces.push(&face);
                    }
                }
            }
        }
        if matching_faces.len() > 0 {
            Ok(matching_faces)
        } else {
            Err(NoHit::new("No hit"))
        }
    }
    fn calculate_hit(&self, ray: &Ray) -> Result<&Polygon, NoHit> {
        // todo: add filtering
        for polyhedron in self.polyhedrons.iter() {
            let faces = &polyhedron.faces;
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
    transmissivity: f64,
    reflectivity: f64,
}

#[pymethods]
impl Polyhedron {
    // constructor parses .obj file into RawGeometry struct,
    // and then converts that into Polyhedron and Polygon structs
    #[new]
    fn new(
        filename: &str,
        index: f64,
        pos: Vec<f64>,
        rot: Vec<Vec<f64>>,
        transmissivity: f64,
        reflectivity: f64,
    ) -> Polyhedron {
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
        new_obj.convert(index, pos, rot, transmissivity, reflectivity)
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
        for (edgenum, result) in results.clone().into_iter().enumerate() {
            if result.is_nan() {
                inside_flag[edgenum] = false;
            } else if result >= 0.0 {
                inside_flag[edgenum] = true;
            }
        }
        is_inside_on_all_sides = inside_flag.into_iter().all(|x| x);
        is_inside_on_all_sides
    }
    fn distance_from_ray(&self, ray: &Ray) -> f64 {
        let intersection = self.intersection(ray);
        let distance = (ray.pos - intersection).norm();
        distance
    }
    fn calculate_next_rays(
        &self,
        incident_ray: &Ray,
        n: f64,
        transmissivity: f64,
        reflectivity: f64,
    ) -> Vec<Ray> {
        // reflected ray
        let mut exiting: bool = false;
        let mut n_relative: f64 = 1.0 / n;
        let mut surface_normal: na::Vector3<f64> = self.normal;
        if incident_ray.dir.dot(&self.normal) > 0.0 {
            exiting = true;
            n_relative = n;
            surface_normal = -surface_normal;
        }
        let mut reflected_ray = Ray {
            power: incident_ray.power * reflectivity,
            index: 0.0,
            pos: self.intersection(incident_ray),
            dir: incident_ray.dir,
            color: incident_ray.color,
            last_ray: false,
            first_ray: false,
        };
        let mut refracted_ray = Ray {
            power: incident_ray.power * transmissivity,
            index: 0.0,
            pos: self.intersection(incident_ray),
            dir: incident_ray.dir,
            color: incident_ray.color,
            last_ray: false,
            first_ray: false,
        };
        let terminal_ray = Ray {
            power: incident_ray.power,
            index: 0.0,
            pos: self.intersection(incident_ray),
            dir: incident_ray.dir,
            color: incident_ray.color,
            last_ray: true,
            first_ray: false,
        };
        let reflected_dir = reflection_matrix(surface_normal) * incident_ray.dir;
        let refracted_dir = (-n_relative * surface_normal.dot(&incident_ray.dir)
            - (1.0
                - n_relative.powf(2.0) * (1.0 - surface_normal.dot(&incident_ray.dir).powf(2.0)))
            .sqrt())
            * surface_normal
            + n_relative * incident_ray.dir;

        reflected_ray.dir = reflected_dir;
        refracted_ray.dir = refracted_dir;

        if exiting {
            reflected_ray.index = n;
            refracted_ray.index = 1.0;
        } else {
            reflected_ray.index = 1.0;
            refracted_ray.index = n;
        }

        let mut return_rays: Vec<Ray> = vec![];

        for ray in vec![reflected_ray, refracted_ray].into_iter() {
            if ray.power > 0.1 && ray.last_ray == false {
                return_rays.push(ray);
            }
        }
        if return_rays.len() == 0 && incident_ray.last_ray == false {
            return_rays.push(terminal_ray);
        }
        return_rays
    }
}

#[pyclass(name=light)]
#[derive(Clone, Debug)]
struct Light {
    pos: na::Vector3<f64>,
    power: f64,
    color: na::Vector3<f64>,
}

#[pymethods]
impl Light {
    #[new]
    fn new(pos: Vec<f64>, power: f64, color: Vec<f64>) -> Light {
        Light {
            power: power,
            pos: na::Vector3::from_vec(pos),
            color: na::Vector3::from_vec(color),
        }
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
    m.add_class::<Light>();
    Ok(())
}
