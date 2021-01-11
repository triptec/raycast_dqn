use std::{io, thread};
use std::io::Read;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::mpsc;
use std::time::Instant;

use clap::Clap;
use csv::Writer;
use moving_avg::MovingAverage;
use rand::{Rng, SeedableRng, thread_rng};
use rand::rngs::StdRng;
use sdl2::pixels::Color;
use serde::Serialize;
use tch::{
    Device,
    kind::{DOUBLE_CPU, FLOAT_CPU, INT64_CPU},
    Kind::Float,
    nn,
    nn::OptimizerConfig,
    Tensor,
};

use input::Input;
use ml::ReplayBuffer;
use sandbox::env::Env;

use crate::ml::Model_a2c;
use crate::renderer::Renderer;

mod renderer;
mod ml;
mod input;

pub fn main() {
    let opts: Opts = Opts::parse();
    let mut rng: StdRng = SeedableRng::seed_from_u64(opts.RANDOM_SEED);
    let stdin_channel = spawn_stdin_channel();
    let stats_file = match opts.STATS_FILE {
        Some(p) => p,
        None => format!("{}.csv", chrono::Utc::now().to_rfc3339())
    };
    let mut wtr = Writer::from_path(stats_file).unwrap();


    let mut env_agent = sandbox::agent::Agent::new(
        opts.AGENT_SPEED,
        opts.AGENT_RAY_COUNT,
        opts.AGENT_FOV,
        opts.AGENT_VISIBILITY,
        opts.AGENT_MAX_AGE,
        opts.AGENT_FOOD,
    );
    let mut env = Env::new(
        opts.ENV_FILE,
        opts.REWARD_WALL_HIT,
        opts.REWARD_WALL_PROXIMITY,
        opts.REWARD_TARGET_FOUND,
        opts.REWARD_TARGET_BEARING_MULT,
        opts.REWARD_TARGET_STEPS_MULT,
        opts.WALL_PROXIMITY,
    );
    env.add_agent(env_agent);
    let mut renderer = Renderer::new(env.scalex, env.scaley);
    println!("action space: {}", env.action_space());
    println!("observation space: {}", env.observation_space());

    let num_obs = env.observation_space() as usize;
    let num_actions = env.action_space() as usize;

    let mut model = Model_a2c::new(num_obs, num_actions, opts.GAMMA, opts.TAU,opts.ACTOR_LEARNING_RATE, opts.ACTOR_LAYERS);

    let mut replay_buffer =
        ReplayBuffer::new(opts.REPLAY_BUFFER_CAPACITY, num_obs, num_actions);

    let mut epsilon = opts.EPSILON;
    let mut epsilon_min = opts.MIN_EPSILON;
    let mut epsilon_decay = opts.EPSILON_DECAY;
    let mut episode = 0;
    let mut total_steps = 0;
    let mut total_targets: i32 = 0;
    let mut total_rewards: f64 = 0.0;
    let mut target_step_avg_100 = MovingAverage::new(100);
    let mut target_avg_100 = MovingAverage::new(100);
    let mut steps_sec_avg_100 = MovingAverage::new(100);
    let mut max_target_avg_100: f64 = 0.0;
    let mut evaluate = false;
    let mut output = true;
    let mut render = false;
    'running: for _ in 0..opts.MAX_EPISODES as i32 {
        let input = match renderer.get_input() {
            Input::None => get_input(&stdin_channel),
            i => i
        };
        match input {
            Input::None => {}
            Input::Quit => {
                println!("Quit");
                break 'running;
            }
            Input::ToggleRender => {
                println!("Toggle render");
                render = !render;
            }
            Input::ToggleOutput => {
                println!("Toggle output");
                output = !output;
                println!("Output: {}", output);
            }
            Input::ToggleEvaluate => {
                println!("Toggle evaluate");
                evaluate = !evaluate;
            }
        }
        let start = Instant::now();
        let mut obs = Tensor::zeros(&[num_obs as _], FLOAT_CPU);
        env.reset(0, 1.0f64);
        let mut episode_rewards = 0.0;
        let mut episode_steps = 0;

        loop {
            let actions_tensor = if evaluate {
                tch::no_grad(|| model.actor.forward(&obs))
            } else {
                if rng.gen_range(0.0, 1.0) < epsilon {
                    let actions: Vec<f64> = (0..num_actions).map(|_| {rng.gen_range(-1.0, 1.0)}).collect();
                    Tensor::of_slice(&actions).totype(Float)
                } else {
                    tch::no_grad(|| model.actor.forward(&obs))
                }
            };
            let action = i32::from(&actions_tensor.argmax(-1, false));
            let (state, reward, done) = env.step(action, 0);
            if render {
                render_env(&mut env, &mut renderer);
            }

            let state_t = Tensor::of_slice(&state).totype(Float);
            if !evaluate {
                replay_buffer.push(&obs, &actions_tensor.into(), &reward.into(), &state_t);
            }
            if done {
                break;
            }
            obs = state_t;
            episode_steps += 1;
            episode_rewards += reward;
        }
        let episode_targets = env.agents.get(0).unwrap().targets_found;
        if !evaluate {
            total_targets += episode_targets;
            total_rewards += episode_rewards;
            total_steps += episode_steps;
            episode += 1;
        }
        let targets_per_step = match total_targets {
            0 => 0f64,
            x => x as f64 / total_steps as f64,
        };
        let targets_per_episode = match total_targets {
            0 => 0f64,
            x => x as f64 / episode as f64,
        };

        let tmp_target_step_avg_100 = target_step_avg_100.feed(episode_targets as f64/episode_steps as f64);
        let tmp_target_avg_100 = target_avg_100.feed(episode_targets as f64);
        max_target_avg_100 = max_target_avg_100.max(tmp_target_avg_100);

        let record = StatsRow {
            episode,
            training_steps: episode * opts.TRAINING_BATCH_SIZE as i32,
            episode_steps,
            episode_targets,
            episode_rewards,
            total_steps,
            total_targets,
            target_step_avg_100: tmp_target_step_avg_100,
            target_avg_100: tmp_target_avg_100,
            max_target_avg_100: max_target_avg_100,
            epsilon,
        };
        if evaluate {

        } else {
            epsilon *= epsilon_decay;
            if epsilon < epsilon_min {
                epsilon = epsilon_min;
            }
            for _ in 0..opts.TRAINING_ITERATIONS {
                model.train(&mut replay_buffer,opts.TRAINING_BATCH_SIZE);
            }
        }

        let tmp_steps_sec_avg_100 = steps_sec_avg_100.feed(episode_steps as f64 / start.elapsed().as_secs_f64());
        if output {
            println!("{:#?}", &record);
            println!("steps/sec: {}", tmp_steps_sec_avg_100);
        }
        wtr.serialize(record).unwrap();
        wtr.flush().unwrap();
    }
}

fn render_env(env: &mut Env, renderer: &mut Renderer) {
    renderer.render = true;
    renderer.init();
    renderer.clear();
    renderer.render_line_strings(
        &env.line_strings.iter().collect(),
        Color::RGB(0, 255, 0),
        &env.agents.get(0).unwrap().position,
    );
    renderer.render_points(
        &env.possible_targets,
        Color::RGB(255, 0, 255),
        &env.agents.get(0).unwrap().position,
    );
    renderer.render_rays(
        &env.agents.get(0).unwrap().rays,
        Color::RGB(0, 0, 255),
        &env.agents.get(0).unwrap().position,
    );
    renderer.present();
    renderer.render = false;
}

fn get_input(stdin_channel: &Receiver<String>) -> Input {
    match stdin_channel.try_recv() {
        Ok(key) => {
            println!("Received: {}", key);
            match &key.chars().nth(0).unwrap() {
                'q' => {
                    Input::Quit
                },
                'r' => {
                    Input::ToggleRender
                },
                'e' => {
                    Input::ToggleEvaluate
                },
                'o' => {
                    Input::ToggleOutput
                },
                _ => { Input::None}
            }
        },
        Err(TryRecvError::Empty) => {Input::None},
        Err(TryRecvError::Disconnected) => panic!("Channel disconnected"),
    }
}

fn spawn_stdin_channel() -> Receiver<String> {
    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || loop {
        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer).unwrap();
        tx.send(buffer).unwrap();
    });
    rx
}


#[derive(Clap)]
#[clap(version = "1.0")]
struct Opts {
    /// Actor layers
    #[clap(long, default_value = "512")]
    ACTOR_LAYERS: Vec<i64>,
    /// Critic layers
    #[clap(long, default_value = "512")]
    CRITIC_LAYERS: Vec<i64>,
    /// The geometric env file.
    #[clap(short, long, default_value = "data/gavle.json")]
    ENV_FILE: String,
    /// Statistics output file.
    #[clap(short, long)]
    STATS_FILE: Option<String>,
    /// Should enable rendering.
    #[clap(long, default_value = "true", parse(try_from_str))]
    RENDER: bool,
    /// The impact of the q value of the next state on the current state's q value.
    #[clap(long, default_value = "0.997")]
    GAMMA: f64,
    /// The weight for updating the target networks.
    #[clap(long, default_value = "0.005")]
    TAU: f64,
    /// The capacity of the replay buffer used for sampling training data.
    #[clap(long, default_value = "1000000")]
    REPLAY_BUFFER_CAPACITY: usize,
    /// The training batch size for each training iteration.
    #[clap(long, default_value = "256")]
    TRAINING_BATCH_SIZE: usize,
    /// The total number of episodes.
    #[clap(long, default_value = "20000")]
    MAX_EPISODES: usize,
    /// The maximum length of an episode.
    #[clap(long, default_value = "1000")]
    EPISODE_LENGTH: usize,
    /// The number of training iterations after one episode finishes.
    #[clap(long, default_value = "1")]
    TRAINING_ITERATIONS: usize,
    /// Ornstein-Uhlenbeck process parameter MU.
    #[clap(long, default_value = "0.0")]
    MU: f64,
    /// Ornstein-Uhlenbeck process parameter THETA.
    #[clap(long, default_value = "0.15")]
    THETA: f64,
    /// Ornstein-Uhlenbeck process parameter SIGMA.
    #[clap(long, default_value = "0.1")]
    SIGMA: f64,
    /// Random noise process parameter EPSILON
    #[clap(long, default_value = "1.0")]
    EPSILON: f64,
    /// Random noise process parameter MIN_EPSILON
    #[clap(long, default_value = "0.01")]
    MIN_EPSILON: f64,
    /// Random noise process parameter EPSILON_DECAY
    #[clap(long, default_value = "0.9999")]
    EPSILON_DECAY: f64,
    /// The leadning rate of the Actor
    #[clap(long, default_value = "0.0005")]
    ACTOR_LEARNING_RATE: f64,
    /// The leadning rate of the Critic
    #[clap(long, default_value = "0.001")]
    CRITIC_LEARNING_RATE: f64,
    /// Random seed
    #[clap(long, default_value = "1")]
    RANDOM_SEED: u64,
    /// Agent speed
    #[clap(long, default_value = "0.007")]
    AGENT_SPEED: f64,
    /// Agent ray count
    #[clap(long, default_value = "29")]
    AGENT_RAY_COUNT: i32,
    /// Agent field of view
    #[clap(long, default_value = "130.0")]
    AGENT_FOV: f64,
    /// Agent visibility (how for it can see)
    #[clap(long, default_value = "0.6")]
    AGENT_VISIBILITY: f64,
    /// Agent food
    #[clap(long, default_value = "1000")]
    AGENT_FOOD: i32,
    /// Agent max age
    #[clap(long, default_value = "10000")]
    AGENT_MAX_AGE: i32,
    /// Agent position ticker
    #[clap(long, default_value = "70")]
    AGENT_POSITION_TICKER: i32,

    /// Env reward for hitting a wall
    #[clap(long, default_value = "-7.0")]
    REWARD_WALL_HIT: f64,
    /// Env reward if agent is within proximity to a wall
    #[clap(long, default_value = "-2.0")]
    REWARD_WALL_PROXIMITY: f64,
    /// Env reward for target found
    #[clap(long, default_value = "7.0")]
    REWARD_TARGET_FOUND: f64,
    /// Env reward multiplier for target bearing
    #[clap(long, default_value = "0.33333333")]
    REWARD_TARGET_BEARING_MULT: f64,
    /// Env reward multiplier for target steps
    #[clap(long, default_value = "0.33333333")]
    REWARD_TARGET_STEPS_MULT: f64,
    /// Env wall proximity
    #[clap(long, default_value = "10.0")]
    WALL_PROXIMITY: f64,
}

#[derive(Serialize, Debug)]
struct StatsRow {
    episode: i32,
    training_steps: i32,
    episode_steps: i32,
    episode_targets: i32,
    episode_rewards: f64,
    total_steps: i32,
    total_targets: i32,
    target_step_avg_100: f64,
    target_avg_100: f64,
    max_target_avg_100: f64,
    epsilon: f64,
}
