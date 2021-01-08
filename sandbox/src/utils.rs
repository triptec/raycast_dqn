use crate::ray::Ray;

use geo::bearing::Bearing;
use geo::closest_point::ClosestPoint;
use geo::euclidean_distance::EuclideanDistance;
use geo::intersects::Intersects;
use geo::map_coords::MapCoordsInplace;
use geo::{
    Closest, Coordinate, Geometry, GeometryCollection, Line, LineString, Point, Polygon, Rect,
};
use geojson::{quick_collection, GeoJson};
use line_intersection::{LineInterval, LineRelation};
use rayon::prelude::*;
use std::fs;

fn load_json(path: String) -> GeometryCollection<f64> {
    let path = String::from(format!("{}", path));
    let geojson_str = fs::read_to_string(path).unwrap();
    let geojson = geojson_str.parse::<GeoJson>().unwrap();
    quick_collection(&geojson).unwrap()
}

pub fn import_geometry(
    path: String,
) -> (Vec<LineString<f64>>, Vec<Point<f64>>, f64, f64, f64, f64) {
    let collection = load_json(path);
    let (mut lines, mut targets) = env_from_geometry(collection);
    let (xmin, ymin, _xmax, _ymax) = calculate_scales(&lines);
    let scalex = 0.02007599999999954 / 10.0;
    let scaley = 0.008532999999999902 / 10.0;
    scale_geometry(scalex, scaley, xmin, ymin, &mut lines, &mut targets);
    (lines, targets, scalex, scaley, xmin, ymin)
}

pub fn scale_geometry(
    scalex: f64,
    scaley: f64,
    xmin: f64,
    ymin: f64,
    lines: &mut Vec<LineString<f64>>,
    targets: &mut Vec<Point<f64>>,
) {
    for point in targets.iter_mut() {
        point.map_coords_inplace(|&(x, y)| ((x - xmin) / scalex, (y - ymin) / scaley));
    }
    for line in lines.iter_mut() {
        line.map_coords_inplace(|&(x, y)| ((x - xmin) / scalex, (y - ymin) / scaley));
    }
}

pub fn env_from_geometry(
    mut collection: GeometryCollection<f64>,
) -> (Vec<LineString<f64>>, Vec<Point<f64>>) {
    let mut lines: Vec<LineString<_>> = vec![];
    let mut points: Vec<Point<_>> = vec![];
    for i in collection.iter_mut() {
        match i {
            Geometry::Point(ref x) => points.push(x.clone()),
            Geometry::Line(_) => {}
            Geometry::LineString(ref x) => lines.push(x.clone()),
            Geometry::Polygon(ref x) => lines.push(x.exterior().clone()),
            Geometry::MultiPoint(_) => {}
            Geometry::MultiLineString(_) => {}
            Geometry::MultiPolygon(_) => {}
            Geometry::GeometryCollection(_) => {}
            Geometry::Rect(_) => {}
            Geometry::Triangle(_) => {}
        }
    }
    (lines, points)
}

pub fn calculate_scales(lines: &Vec<LineString<f64>>) -> (f64, f64, f64, f64) {
    let mut xmin: f64 = 999.9;
    let mut ymin: f64 = 999.9;
    let mut xmax: f64 = 0.0;
    let mut ymax: f64 = 0.0;
    for line in lines.iter() {
        for point in line.points_iter() {
            xmin = xmin.min(point.x());
            ymin = ymin.min(point.y());
            xmax = xmax.max(point.x());
            ymax = ymax.max(point.y());
        }
    }
    //let scalex = 1.0 * (xmax - xmin);
    //let scaley = 1.0 * (ymax - ymin);
    //println!("scale x {}", scalex);
    //println!("scale y {}", scaley);
    //(scalex, scaley, xmin, ymin, xmax, ymax)
    (xmin, ymin, xmax, ymax)
}

pub fn cull_line_strings<'a>(
    rays_bb: &Rect<f64>,
    line_strings: &'a Vec<LineString<f64>>,
    _origin_position: Point<f64>,
) -> Vec<&'a LineString<f64>> {
    let bbox = rays_bb.to_polygon();
    let mut intersecting_line_strings = vec![];
    for line_string in line_strings.iter() {
        if bbox.intersects(line_string) {
            intersecting_line_strings.push(line_string)
        }
    }
    intersecting_line_strings
}

#[inline(always)]
pub fn min_max_point(
    coordinate: &Coordinate<f64>,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) -> (f64, f64, f64, f64) {
    (
        min_x.min(coordinate.x),
        min_y.min(coordinate.y),
        max_x.max(coordinate.x),
        max_y.max(coordinate.y),
    )
}

#[inline(always)]
pub fn min_max_lines(lines: &Vec<Line<f64>>, shared_start: bool) -> (f64, f64, f64, f64) {
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;

    if shared_start {
        min_x = lines[0].start.x;
        min_y = lines[0].start.y;
        max_x = min_x;
        max_y = min_y;
        for line in lines.iter() {
            max_x = max_x.max(line.end.x);
            max_y = max_y.max(line.end.y);
            min_x = min_x.min(line.end.x);
            min_y = min_y.min(line.end.y);
        }
    } else {
        for line in lines.iter() {
            max_x = max_x.max(line.start.x).max(line.end.x);
            max_y = max_y.max(line.start.y).max(line.end.y);
            min_x = min_x.min(line.start.x).min(line.end.x);
            min_y = min_y.min(line.start.y).min(line.end.y);
        }
    }
    (min_x, min_y, max_x, max_y)
}

#[inline(always)]
pub fn might_intersect_line(
    line: &Line<f64>,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) -> bool {
    if line.start.x > max_x && line.end.x > max_x {
        return false;
    }
    if line.start.y > max_y && line.end.y > max_y {
        return false;
    }
    if line.start.x < min_x && line.end.x < min_x {
        return false;
    }
    if line.start.y < min_y && line.end.y < min_y {
        return false;
    }
    true
}

#[inline(always)]
pub fn might_intersect_line_string(
    line_string: &LineString<f64>,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) -> bool {
    for line in line_string.lines() {
        if might_intersect_line(&line, min_x, min_y, max_x, max_y) {
            return true;
        }
    }
    false
}

pub fn cull_line_strings_precull<'a>(
    rays_bb: &Rect<f64>,
    line_strings: &'a Vec<LineString<f64>>,
    _origin_position: Point<f64>,
) -> Vec<&'a LineString<f64>> {
    let (min_x, min_y) = rays_bb.min().x_y();
    let (max_x, max_y) = rays_bb.max().x_y();
    let bbox = rays_bb.to_polygon();
    let intersecting_line_strings = line_strings.par_iter().filter(|line_string| {
        might_intersect_line_string(line_string, min_x, min_y, max_x, max_y)
            && bbox.intersects(line_string.clone())
    });
    intersecting_line_strings.collect()
}

pub fn find_intersections_par(
    rays: &mut Vec<Ray>,
    line_strings: &Vec<&LineString<f64>>,
    origin_position: Point<f64>,
) {
    rays.par_iter_mut()
        .for_each(|ray| find_intersections(ray, line_strings, origin_position));
}

pub fn intersects(ray: &Ray, line_strings: &Vec<&LineString<f64>>) -> bool {
    for line in line_strings {
        let intersections = intersections(&ray.line_string, line);
        if intersections.len() > 0 {
            return true;
        }
    }
    return false;
}

pub fn find_intersections(
    ray: &mut Ray,
    line_strings: &Vec<&LineString<f64>>,
    origin_position: Point<f64>,
) {
    for line in line_strings.iter() {
        let intersections = intersections(&ray.line_string, line);
        for intersection in intersections.iter() {
            let length = intersection.euclidean_distance(&origin_position);
            if length < ray.length {
                ray.length = length;
                ray.line_string = LineString(vec![ray.line_string.0[0], intersection.0])
            }
        }
    }
}
// Would be better to use agent_direction instead of "agent_step_ray_end"
pub fn relative_bearing_to_target(
    agent_position: Point<f64>,
    agent_step_ray_end: Point<f64>,
    target_position: Point<f64>,
) -> f64 {
    let target_bearing = agent_position.bearing(target_position);
    let step_bearing = agent_position.bearing(agent_step_ray_end);
    let d = target_bearing - step_bearing;
    return (if d > 180.0 {
        (-180.0 + d - 180.0)
    } else if d < -180.0 {
        (180.0 + d + 180.0)
    } else {
        d
    })
    .to_radians();
}

pub fn intersections(
    linestring1: &LineString<f64>,
    linestring2: &LineString<f64>,
) -> Vec<Point<f64>> {
    let mut intersections = vec![];
    if linestring1.0.is_empty() || linestring2.0.is_empty() {
        return vec![];
    }
    for a in linestring1.lines() {
        for b in linestring2.lines() {
            let a_li = LineInterval::line_segment(a);
            let b_li = LineInterval::line_segment(b);
            match a_li.relate(&b_li) {
                LineRelation::DivergentIntersecting(x) => intersections.push(x),
                _ => {}
            }
        }
    }
    intersections
}

pub fn closest_of<C, I>(iter: I, p: Point<f64>) -> Option<Point<f64>>
where
    I: IntoIterator<Item = C>,
    C: ClosestPoint<f64>,
{
    let mut best = Closest::Indeterminate;

    for line_segment in iter {
        let got = line_segment.closest_point(&p);
        best = got.best_of_two(&best, p);
    }

    match best {
        Closest::Indeterminate => None,
        Closest::Intersection(p) => Some(p),
        Closest::SinglePoint(p) => Some(p),
    }
}

#[cfg(test)]
mod tests {
    use crate::ray::Ray;
    use crate::utils;
    use geo::Point;
    use test::Bencher;

    #[bench]
    fn test_culling_obstacles(b: &mut Bencher) {
        let position = Point::new(0.5, 0.5);
        let (mut rays, rays_bb) = Ray::generate_rays(180.0, 0.4, 0.3, 0.1, position);
        let (line_strings, points, _scalex, _scaley) =
            utils::import_geometry("../data/obstacles.json".into());
        b.iter(|| utils::cull_line_strings(&rays_bb, &line_strings, position));
    }

    #[bench]
    fn test_culling_obstacles_preculling(b: &mut Bencher) {
        let position = Point::new(0.5, 0.5);
        let (mut rays, rays_bb) = Ray::generate_rays(180.0, 0.4, 0.3, 0.1, position);
        let (line_strings, points, _scalex, _scaley) =
            utils::import_geometry("../data/obstacles.json".into());
        b.iter(|| utils::cull_line_strings_precull(&rays_bb, &line_strings, position));
    }

    #[bench]
    fn test_culling_polygons(b: &mut Bencher) {
        let position = Point::new(0.5, 0.5);
        let (mut rays, rays_bb) = Ray::generate_rays(180.0, 0.4, 0.3, 0.1, position);
        let (line_strings, points, _scalex, _scaley) =
            utils::import_geometry("../data/polygons.json".into());
        b.iter(|| utils::cull_line_strings(&rays_bb, &line_strings, position));
    }

    #[bench]
    fn test_culling_polygons_preculling(b: &mut Bencher) {
        let position = Point::new(0.5, 0.5);
        let (mut rays, rays_bb) = Ray::generate_rays(180.0, 0.4, 0.3, 0.1, position);
        let (line_strings, points, _scalex, _scaley) =
            utils::import_geometry("../data/polygons.json".into());
        b.iter(|| utils::cull_line_strings_precull(&rays_bb, &line_strings, position));
    }
}
