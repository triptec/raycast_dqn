use crate::agent::Agent;
use crate::ray::Ray;
use crate::utils;
use geo::euclidean_distance::EuclideanDistance;
use geo::{LineString, Point};
use rand;
use rand::prelude::IteratorRandom;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::HashMap;
use std::f32::consts::PI;

pub struct Env {
    pub line_strings: Vec<LineString<f64>>,
    pub targets: Vec<Point<f64>>,
    pub possible_targets: Vec<Point<f64>>,
    pub scalex: f64,
    pub scaley: f64,
    pub agents: Vec<Agent>,
    pub original_targets: Vec<Point<f64>>,
    pub reward_wall_hit: f64,
    pub reward_wall_proximity: f64,
    pub reward_target_found: f64,
    pub reward_target_bearing_mult: f64,
    pub reward_target_steps_mult: f64,
    pub wall_proximity: f64,
}

impl Env {
    pub fn new(
        path: String,
        reward_wall_hit: f64,
        reward_wall_proximity: f64,
        reward_target_found: f64,
        reward_target_bearing_mult: f64,
        reward_target_steps_mult: f64,
        wall_proximity: f64,
    ) -> Self {
        let (line_strings, targets, scalex, scaley, xmin, ymin) = utils::import_geometry(path);
        Env {
            original_targets: targets.iter().copied().collect(),
            line_strings,
            possible_targets: targets.iter().copied().collect(),
            targets,
            scalex,
            scaley,
            agents: vec![],
            reward_wall_hit,
            reward_wall_proximity,
            reward_target_found,
            reward_target_bearing_mult,
            reward_target_steps_mult,
            wall_proximity,
        }
    }

    pub fn add_agent(&mut self, mut agent: Agent) {
        agent.add_env_info(
            self.targets
                .choose(&mut rand::thread_rng())
                .unwrap()
                .clone(),
            self.line_strings.clone(),
        );
        self.agents.push(agent);
    }

    pub fn action_space(&self) -> usize {
        self.agents.get(0).unwrap().action_space.len()
    }

    pub fn observation_space(&self) -> usize {
        /*
        relative_bearing_to_target,
        steps_to_target,
        past_position_bearing,
        rays
        */
        3 + self.agents.get(0).unwrap().ray_count as usize
    }

    pub fn step(&mut self, action: i32, a: i32) -> (Vec<f64>, f64, bool) {
        let direction_change = self.agents[a as usize]
            .action_space
            .get(action as usize)
            .unwrap()
            .clone();
        let mut reward = 0.0;
        let step_ray = Ray::new(
            direction_change,
            self.agents[a as usize].speed,
            self.agents[a as usize].direction,
            self.agents[a as usize].position,
            false,
            0.0,
        );
        if utils::intersects(&step_ray, &self.line_strings.iter().collect()) {
            let state = self.agents[a as usize].last_state.iter().copied().collect();
            reward = self.reward_wall_hit;
            self.agents[a as usize].active = false;
            return (state, reward, true);
        }

        let proximity_ray = Ray::new(
            direction_change,
            self.agents[a as usize].speed * self.wall_proximity,
            self.agents[a as usize].direction,
            self.agents[a as usize].position,
            false,
            0.0,
        );
        if utils::intersects(&proximity_ray, &self.line_strings.iter().collect()) {
            reward = self.reward_wall_proximity;
        }

        self.agents[a as usize].age += 1.0;
        self.agents[a as usize].food -= 1.0;
        self.agents[a as usize].step(action as usize);
        let (mut state, closest_target) = self.get_state(a, step_ray);
        self.agents[a as usize].closest_target = closest_target;
        // Target
        reward = reward - state[0].abs() / 3.0; // self.reward_target_bearing_mult;
        reward = reward - state[1] / 3.0; //* self.reward_target_steps_mult; // steps_to_target / 3
                                          // Past position
                                          //reward = reward - (1.0-state[2].abs()) / 20.0; // relative bearing to past position / 20
                                          //reward = reward - (1.0-state[3]) / 20.0; //
        if state[1] * 1000.0 < 10.0 {
            state = self.agents[a as usize].last_state.iter().copied().collect();
            reward = self.reward_target_found;
            self.possible_targets
                .retain(|t| t.x_y() != closest_target.x_y());
            self.agents[a as usize].collect_target(closest_target, self.targets.len() as i32);
        }
        self.agents[a as usize].last_state = state.iter().copied().collect();
        //dbg!(&reward);
        return (state, reward, !self.agents[a as usize].active);
    }

    pub fn get_state(&mut self, a: i32, mut step_ray: Ray) -> (Vec<f64>, Point<f64>) {
        let step_ray = Ray::new(
            0.0,
            self.agents[a as usize].speed,
            self.agents[a as usize].direction,
            self.agents[a as usize].position,
            false,
            0.0,
        );
        let mut state = vec![];
        let closest_target = utils::closest_of(
            self.possible_targets.iter(),
            self.agents[a as usize].position,
        )
        .unwrap();
        let relative_bearing_to_target = utils::relative_bearing_to_target(
            self.agents[a as usize].position,
            step_ray.line.end_point(),
            closest_target,
        );
        state.push(relative_bearing_to_target / 3.14159);

        let distance_to_target = self.agents[a as usize]
            .position
            .euclidean_distance(&closest_target);
        let steps_to_target = (distance_to_target / self.agents[a as usize].speed) / 1000.0;
        state.push(steps_to_target);
        state.push(self.agents[a as usize].past_position_bearing / 3.14159);
        let mut ray_lengths = self.agents[a as usize]
            .rays
            .iter()
            .map(|r| r.length / r.max_length)
            .collect();
        state.append(&mut ray_lengths);
        return (state, closest_target);
    }

    pub fn reset(&mut self, agent_index: i32, epsilon: f64) {
        let mut new_targets = vec![];
        let mut take_targets = self.original_targets.len() as f64;
        if self.original_targets.len() as f64 * (epsilon + epsilon) + 10.0
            < self.original_targets.len() as f64
        {
            take_targets = self.original_targets.len() as f64 * (epsilon + epsilon) + 10.0;
        }
        self.original_targets
            .iter()
            .choose_multiple(&mut rand::thread_rng(), take_targets as usize)
            .iter()
            .for_each(|p| {
                new_targets.push(p.clone().clone());
            });
        self.targets = new_targets.clone();
        self.targets.shuffle(&mut rand::thread_rng());
        let start = self
            .targets
            .choose(&mut rand::thread_rng())
            .unwrap()
            .clone();
        self.possible_targets = self
            .targets
            .iter()
            .filter(|t| t.x_y() != start.x_y())
            .copied()
            .collect();
        //self.agents[agent_index as usize] = Agent::new(start, self.line_strings.clone(), self.max_steps + ((1.0 - epsilon) * 1000.0) as i32);
        self.agents[agent_index as usize].reset(start);
        self.agents[agent_index as usize].cast_rays();
    }
}
#[cfg(test)]
mod tests {
    use geo::bearing::Bearing;
    use geo::Point;

    #[test]
    fn test_bearing() {
        let agent_position = Point::new(0.0, 0.0);
        let agent_step_ray_end = Point::new(-2.0, -2.0);
        let target_position = Point::new(2.0, 0.0);

        let target_bearing = agent_position.bearing(target_position);
        let step_bearing = agent_position.bearing(agent_step_ray_end);
        dbg!(target_bearing);
        dbg!(step_bearing);
        let d = target_bearing - step_bearing;
        dbg!(d);
        if d > 180.0 {
            dbg!(-180.0 + d - 180.0);
        } else if d < -180.0 {
            dbg!(180.0 + d + 180.0);
        }
    }
}
