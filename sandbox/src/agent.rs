use crate::ray::Ray;
use crate::utils;
use geo::euclidean_distance::EuclideanDistance;
use geo::map_coords::MapCoordsInplace;
use geo::{LineString, Point, Rect};
use rand::Rng;
use std::collections::HashMap;
pub struct Agent {
    pub speed: f64,
    pub direction: f64,
    pub ray_count: i32,
    pub fov: f64,
    pub visibility: f64,
    pub position: Point<f64>,
    pub rays: Vec<Ray>,
    pub rays_bb: Rect<f64>,
    pub age: f64,
    pub closest_target: Point<f64>,
    pub max_age: f64,
    pub food: f64,
    pub food_start: f64,
    pub active: bool,
    pub action_space: Vec<f64>,
    pub prev_state: Vec<f64>,
    pub collected_targets: Vec<Point<f64>>,
    pub last_state: Vec<f64>,
    pub env_line_strings: Vec<LineString<f64>>,
    pub near_env_line_strings: Vec<LineString<f64>>,
    pub near_zone: Rect<f64>,
    pub recalc: i32,
    pub targets_found: i32,
    pub past_position: Point<f64>,
    pub past_position_distance: f64,
    pub past_position_bearing: f64,
    pub evaluating: bool,
    pub evaluation_score: f64,
}

impl Agent {
    pub fn new(
        speed: f64,
        ray_count: i32,
        fov: f64,
        visibility: f64,
        max_age: i32,
        food: i32,
    ) -> Self {
        Agent {
            speed, // 0.0045
            age: 1.0,
            direction: rand::thread_rng().gen_range(-3.14..=3.14),
            ray_count,
            fov: fov.to_radians(),
            visibility,
            max_age: max_age as f64,
            food: food as f64,
            food_start: food as f64,
            position: Point::new(0.0, 0.0),
            rays: vec![],
            rays_bb: Rect::new(
                (f64::NEG_INFINITY, f64::NEG_INFINITY),
                (f64::INFINITY, f64::INFINITY),
            ),
            collected_targets: vec![],
            targets_found: 0,
            closest_target: Point::new(0.0, 0.0),
            active: true,
            past_position: Point::new(0.0, 0.0),
            past_position_distance: 0.0,
            past_position_bearing: 0.0,
            last_state: vec![],
            action_space: vec![
                -20.0f64.to_radians(),
                -3.0f64.to_radians(),
                0.0f64.to_radians(),
                3.0f64.to_radians(),
                20.0f64.to_radians(),
            ],
            prev_state: vec![],
            env_line_strings: vec![],
            near_env_line_strings: vec![],
            near_zone: Rect::new(
                (f64::NEG_INFINITY, f64::NEG_INFINITY),
                (f64::INFINITY, f64::INFINITY),
            ),
            recalc: 0,
            evaluating: false,
            evaluation_score: 0.0,
        }
    }

    pub(crate) fn add_env_info(
        &mut self,
        mut position: Point<f64>,
        env_line_strings: Vec<LineString<f64>>,
    ) {
        self.env_line_strings = env_line_strings;
        self.collected_targets = vec![position.clone()];
        self.position = position;
        self.past_position = position;
    }

    pub(crate) fn reset(&mut self, mut position: Point<f64>) {
        if !self.evaluating {
            self.direction = rand::thread_rng().gen_range(-3.14..=3.14);
        } else {
            self.direction = 0.0;
        }
        self.position = position.clone();
        self.rays = vec![];
        self.rays_bb = Rect::new(
            (f64::NEG_INFINITY, f64::NEG_INFINITY),
            (f64::INFINITY, f64::INFINITY),
        );
        self.near_zone = Rect::new(
            (f64::NEG_INFINITY, f64::NEG_INFINITY),
            (f64::INFINITY, f64::INFINITY),
        );
        self.near_env_line_strings = vec![];
        self.recalc = 0;
        self.collected_targets = vec![position.clone()];
        self.targets_found = 0;
        self.closest_target = Point::new(0.0, 0.0);
        self.active = true;
        self.age = 1.0;
        self.food = self.food_start;
        self.past_position = position;
        self.past_position_distance = 0.0;
        self.past_position_bearing = 0.0;
        self.last_state = vec![];
        self.prev_state = vec![];
    }

    pub fn cast_rays(&mut self) {
        self.rays.clear();
        let (rays, rays_bb) = Ray::generate_rays(
            self.ray_count as f64,
            self.fov,
            self.visibility,
            self.direction,
            self.position,
        );
        self.rays = rays;
        self.rays_bb = rays_bb;
    }

    pub fn collect_target(&mut self, target: Point<f64>, n_targets: i32) {
        self.food += 100.0;
        if self.evaluating {
            self.food = 2500.0;
            if self.evaluation_score < self.targets_found as f64 {
                self.evaluation_score = self.targets_found as f64;
            }
        }
        self.targets_found = self.targets_found + 1;
        self.collected_targets.push(target);
        if self.collected_targets.len() as i32 == n_targets {
            self.collected_targets = vec![];
        }
        self.past_position = self.position;
    }

    pub fn step(&mut self, action: usize) {
        let step_size = self.speed;
        let direction_change = self.action_space.get(action as usize).unwrap();

        if self.food <= 0.0 {
            self.active = false;
        }
        if self.age > self.max_age && !self.evaluating {
            self.active = false;
        }
        if direction_change == &300.0f64 {
            self.past_position = self.position;
        } else {
            self.direction += direction_change;
            if self.direction > 3.14159 {
                self.direction = self.direction - 3.14159 * 2.0;
            }
            if self.direction < -3.14159 {
                self.direction = self.direction + 3.14159 * 2.0;
            }

            let new_position = Point::new(
                self.position.x() + step_size * self.direction.cos(),
                self.position.y() + step_size * self.direction.sin(),
            );
            self.position = new_position;
        }

        self.past_position_distance = self.position.euclidean_distance(&self.past_position);
        self.past_position_bearing =
            utils::relative_bearing_to_target(self.position, self.position, self.past_position);
        self.cast_rays();
        self.update();
    }

    pub fn update(&mut self) {
        if self.recalc <= 0 {
            let max_dist = (self.speed * 100.0) + self.visibility;
            self.near_zone = Rect::new((self.position.x() - max_dist, self.position.y() - max_dist), (self.position.x() + max_dist, self.position.y() + max_dist));
            self.near_env_line_strings = utils::cull_line_strings_precull(
                &mut self.near_zone,
                &self.env_line_strings,
                self.position,
            ).iter().map(|l|l.to_owned().clone()).collect::<Vec<_>>();
            self.recalc = 100;
        }
        self.recalc -= 1;
        let intersecting_line_strings = utils::cull_line_strings_precull(
            &mut self.rays_bb,
            &self.near_env_line_strings,
            self.position,
        );
        utils::find_intersections_par(&mut self.rays, &intersecting_line_strings, self.position)
    }
}
