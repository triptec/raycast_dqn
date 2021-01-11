/* Deep Deterministic Policy Gradient.
   Continuous control with deep reinforcement learning, Lillicrap et al. 2015
   https://arxiv.org/abs/1509.02971
   See https://spinningup.openai.com/en/latest/algorithms/ddpg.html for a
   reference python implementation.
*/
use clap::Clap;
mod renderer;
use crate::renderer::Renderer;
use csv::Writer;
use moving_avg::MovingAverage;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sandbox::env::Env;
use sdl2::pixels::Color;
use serde::Serialize;
use std::time::Instant;
use tch::{
    kind::{DOUBLE_CPU, FLOAT_CPU, INT64_CPU},
    nn,
    nn::OptimizerConfig,
    Device,
    Kind::Float,
    Tensor,
};

struct OuNoise {
    mu: f64,
    theta: f64,
    sigma: f64,
    state: Tensor,
}

impl OuNoise {
    fn new(mu: f64, theta: f64, sigma: f64, num_actions: usize) -> Self {
        let state = Tensor::ones(&[num_actions as _], FLOAT_CPU);
        Self {
            mu,
            theta,
            sigma,
            state,
        }
    }

    fn sample(&mut self) -> &Tensor {
        let dx = self.theta * (self.mu - &self.state)
            + self.sigma * Tensor::randn(&self.state.size(), FLOAT_CPU);
        self.state += dx;
        &self.state
    }
}

struct MyNoise {
    epsilon: f64,
    decay: f64,
    min_epsilon: f64,
    num_actions: usize,
    rng: StdRng,
    state: Tensor,
}

impl MyNoise {
    fn new(epsilon: f64, decay: f64, min_epsilon: f64, num_actions: usize) -> Self {
        let state = Tensor::zeros(&[num_actions as _], FLOAT_CPU);
        let mut rng: StdRng = SeedableRng::seed_from_u64(1);
        Self {
            epsilon,
            decay,
            min_epsilon,
            num_actions,
            rng,
            state,
        }
    }

    fn sample(&mut self) -> &Tensor {
        self.state = Tensor::zeros(&[self.num_actions as _], FLOAT_CPU);
        if (self.rng.gen_range(0.0, 1.0) < self.epsilon) {
            let action = self.rng.gen_range(0, 5);
            //println!("random action {}", &action);
            let mut zero_vec = vec![0.0; self.num_actions];
            zero_vec[action] = 2.0;
            self.state = Tensor::of_slice(&zero_vec).totype(Float);
        }
        if self.epsilon > self.min_epsilon {
            self.epsilon = self.epsilon * self.decay;
        } else {
            self.epsilon = self.min_epsilon;
        }
        //dbg!(&self.state);
        return &self.state;
    }
}

struct ReplayBuffer {
    obs: Tensor,
    next_obs: Tensor,
    rewards: Tensor,
    actions: Tensor,
    capacity: usize,
    len: usize,
    i: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize, num_obs: usize, num_actions: usize) -> Self {
        Self {
            obs: Tensor::zeros(&[capacity as _, num_obs as _], FLOAT_CPU),
            next_obs: Tensor::zeros(&[capacity as _, num_obs as _], FLOAT_CPU),
            rewards: Tensor::zeros(&[capacity as _, 1], FLOAT_CPU),
            actions: Tensor::zeros(&[capacity as _, num_actions as _], FLOAT_CPU),
            capacity,
            len: 0,
            i: 0,
        }
    }

    fn push(&mut self, obs: &Tensor, actions: &Tensor, reward: &Tensor, next_obs: &Tensor) {
        let i = self.i % self.capacity;
        self.obs.get(i as _).copy_(obs);
        self.rewards.get(i as _).copy_(reward);
        self.actions.get(i as _).copy_(actions);
        self.next_obs.get(i as _).copy_(next_obs);
        self.i += 1;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    fn random_batch(&self, batch_size: usize) -> Option<(Tensor, Tensor, Tensor, Tensor)> {
        if self.len < batch_size * 2 {
            return None;
        }

        let batch_size = batch_size.min(self.len - 1);
        let batch_indexes = Tensor::randint((self.len - 2) as _, &[batch_size as _], INT64_CPU);

        let states = self.obs.index_select(0, &batch_indexes);
        let next_states = self.next_obs.index_select(0, &batch_indexes);
        let actions = self.actions.index_select(0, &batch_indexes);
        let rewards = self.rewards.index_select(0, &batch_indexes);

        Some((states, actions, rewards, next_states))
    }
}

struct Actor {
    var_store: nn::VarStore,
    network: nn::Sequential,
    device: Device,
    num_obs: usize,
    num_actions: usize,
    opt: nn::Optimizer<nn::Adam>,
    learning_rate: f64,
    layers: Vec<i64>,
}

impl Clone for Actor {
    fn clone(&self) -> Self {
        let mut new = Self::new(self.num_obs, self.num_actions, self.learning_rate, self.layers.clone());
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl Actor {
    fn new(num_obs: usize, num_actions: usize, learning_rate: f64, layers: Vec<i64>) -> Self {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let opt = nn::Adam::default()
            .build(&var_store, learning_rate)
            .unwrap();
        let p = &var_store.root();
        let mut network = nn::seq();
        for index in 0..layers.len() {
            if index == 0 {
                network = network
                    .add(nn::linear(p / "first", num_obs as _, layers.get(index).unwrap().clone(), Default::default()))
                    .add_fn(|xs| xs.relu());
            }

            /*
            network = network
                .add(nn::linear(p / format!("l{}", index), layers.get(index).unwrap().clone(), layers.get(index).unwrap().clone(), Default::default()))
                .add_fn(|xs| xs.relu());
            */

            if index == layers.len() - 1 {
                network = network
                    .add(nn::linear(
                        p / "last",
                        layers.get(index).unwrap().clone(),
                        num_actions as _,
                        Default::default(),
                    ))
                    .add_fn(|xs| xs.tanh());
            }
        }
        dbg!(&network);
        Self {
            network,
            device: p.device(),
            num_obs,
            num_actions,
            var_store,
            opt,
            learning_rate,
            layers,
        }
    }

    fn forward(&self, obs: &Tensor) -> Tensor {
        obs.to_device(self.device).apply(&self.network)
    }
}

struct Critic {
    var_store: nn::VarStore,
    network: nn::Sequential,
    device: Device,
    num_obs: usize,
    num_actions: usize,
    opt: nn::Optimizer<nn::Adam>,
    learning_rate: f64,
    layers: Vec<i64>
}

impl Clone for Critic {
    fn clone(&self) -> Self {
        let mut new = Self::new(self.num_obs, self.num_actions, self.learning_rate, self.layers.clone());
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl Critic {
    fn new(num_obs: usize, num_actions: usize, learning_rate: f64, layers: Vec<i64>) -> Self {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let opt = nn::Adam::default().build(&var_store, 1e-3).unwrap();
        let p = &var_store.root();
        let mut network = nn::seq();
        for index in 0..layers.len() {
            if index == 0 {
                network = network
                    .add(nn::linear(p / "clfirst", (num_obs + num_actions) as _, layers.get(index).unwrap().clone(), Default::default()))
                    .add_fn(|xs| xs.relu());
            }

            network = network
                .add(nn::linear(p / format!("cl{}", index), layers.get(index).unwrap().clone(), layers.get(index).unwrap().clone(), Default::default()))
                .add_fn(|xs| xs.relu());


            if index == layers.len() - 1 {
                network = network
                    .add(nn::linear(
                        p / "cllast",
                        layers.get(index).unwrap().clone(),
                        1,
                        Default::default(),
                    ))
            }
        }

        Self {
            network,
            device: p.device(),
            var_store,
            num_obs,
            num_actions,
            opt,
            learning_rate,
            layers,
        }
    }

    fn forward(&self, obs: &Tensor, actions: &Tensor) -> Tensor {
        let xs = Tensor::cat(&[actions.copy(), obs.copy()], 1);
        xs.to_device(self.device).apply(&self.network)
    }
}

fn track(dest: &mut nn::VarStore, src: &nn::VarStore, tau: f64) {
    tch::no_grad(|| {
        for (dest, src) in dest
            .trainable_variables()
            .iter_mut()
            .zip(src.trainable_variables().iter())
        {
            dest.copy_(&(tau * src + (1.0 - tau) * &*dest));
        }
    })
}

struct Agent {
    actor: Actor,
    actor_target: Actor,

    critic: Critic,
    critic_target: Critic,

    replay_buffer: ReplayBuffer,

    ou_noise: OuNoise,
    my_noise: MyNoise,

    train: bool,

    gamma: f64,
    tau: f64,
    pub critic_loss: f64,
    pub actor_loss: f64,
}

impl Agent {
    fn new(
        actor: Actor,
        critic: Critic,
        ou_noise: OuNoise,
        my_noise: MyNoise,
        replay_buffer_capacity: usize,
        train: bool,
        gamma: f64,
        tau: f64,
    ) -> Self {
        let actor_target = actor.clone();
        let critic_target = critic.clone();
        let replay_buffer =
            ReplayBuffer::new(replay_buffer_capacity, actor.num_obs, actor.num_actions);
        Self {
            actor,
            actor_target,
            critic,
            critic_target,
            replay_buffer,
            ou_noise,
            my_noise,
            train,
            gamma,
            tau,
            critic_loss: 0.0,
            actor_loss: 0.0,
        }
    }

    fn actions(&mut self, obs: &Tensor) -> Tensor {
        let mut actions = tch::no_grad(|| self.actor.forward(obs));
        if self.train {
            actions = actions.clamp(-1.0, 0.9999999);
            actions += self.my_noise.sample();
            //actions *= self.my_noise.sample();
            //actions.copy_(self.my_noise.sample());
            actions = actions.clamp(-1.0, 1.0);
        }
        //dbg!(&actions);
        actions
    }

    fn remember(&mut self, obs: &Tensor, actions: &Tensor, reward: &Tensor, next_obs: &Tensor) {
        //dbg!(&obs.kind(), &actions.kind(), &reward.kind(), &next_obs.kind());
        self.replay_buffer.push(obs, actions, reward, next_obs);
    }

    fn train(&mut self, batch_size: usize) {
        let (states, actions, rewards, next_states) =
            match self.replay_buffer.random_batch(batch_size) {
                Some(v) => v,
                _ => return, // We don't have enough samples for training yet.
            };

        let mut q_target = self
            .critic_target
            .forward(&next_states, &self.actor_target.forward(&next_states));
        q_target = rewards.copy() + (self.gamma * q_target).detach();

        let q = self.critic.forward(&states, &actions);

        let diff = q_target.copy() - q.copy();
        let x = diff.argmax(-2, true);
        let index = i64::from(x);
        //dbg!(&states.get(index), &actions.get(index), &rewards.get(index), &next_states.get(index));
        self.remember(&states.get(index).copy(), &actions.get(index).copy(), &rewards.get(index).copy(), &next_states.get(index).copy());
        let critic_loss = (&diff * &diff).mean(Float);
        self.critic_loss = f64::from(&critic_loss);
        self.critic.opt.zero_grad();
        critic_loss.backward();
        self.critic.opt.step();

        let actor_loss = -self
            .critic
            .forward(&states, &self.actor.forward(&states))
            .mean(Float);
        self.actor_loss = f64::from(&actor_loss);
        self.actor.opt.zero_grad();
        actor_loss.backward();
        self.actor.opt.step();

        track(
            &mut self.critic_target.var_store,
            &self.critic.var_store,
            self.tau,
        );
        track(
            &mut self.actor_target.var_store,
            &self.actor.var_store,
            self.tau,
        );
    }
}

pub fn main() {
    let opts: Opts = Opts::parse();
    let mut rng: StdRng = SeedableRng::seed_from_u64(opts.RANDOM_SEED);
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
    if opts.RENDER {
        renderer.init();
    }
    println!("action space: {}", env.action_space());
    println!("observation space: {}", env.observation_space());

    let num_obs = env.observation_space() as usize;
    let num_actions = env.action_space() as usize;
    //let num_actions = 1 as usize;

    let actor = Actor::new(num_obs, num_actions, opts.ACTOR_LEARNING_RATE, opts.ACTOR_LAYERS);
    let critic = Critic::new(num_obs, num_actions, opts.CRITIC_LEARNING_RATE, opts.CRITIC_LAYERS);
    let ou_noise = OuNoise::new(opts.MU, opts.THETA, opts.SIGMA, num_actions);
    let my_noise = MyNoise::new(
        opts.EPSILON,
        opts.EPSILON_DECAY,
        opts.MIN_EPSILON,
        num_actions,
    );
    let mut agent = Agent::new(
        actor,
        critic,
        ou_noise,
        my_noise,
        opts.REPLAY_BUFFER_CAPACITY,
        true,
        opts.GAMMA,
        opts.TAU,
    );
    let mut total_steps = 0;
    let mut last_step = 0;
    let mut total_targets: i32 = 0;
    let mut total_rewards: f64 = 0.0;
    let mut target_step_avg_100 = MovingAverage::new(100);
    let mut target_avg_100 = MovingAverage::new(100);
    let mut max_target_avg_100: f64 = 0.0;
    let start = Instant::now();

    'running: for episode in 0..opts.MAX_EPISODES as i32 {
        if renderer.quit() {
            break 'running;
        }
        let mut obs = Tensor::zeros(&[num_obs as _], FLOAT_CPU);
        env.reset(0, 1.0f64);
        //dbg!(&obs);
        let mut episode_rewards = 0.0;
        loop {
            if (renderer.render) {
                agent.train = false;
            } else {
                agent.train = true;
            }
            let actions = agent.actions(&obs);
            let action = i32::from(&actions.argmax(-1, false));
            let (state, reward, done) = env.step(action, 0);
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
            episode_rewards += reward;
            let state_t = Tensor::of_slice(&state).totype(Float);
            //state_t;
            //dbg!(&state_t);
            if !renderer.render {
                agent.remember(&obs, &actions.into(), &reward.into(), &state_t);
                total_steps += 1;
            }
            if done {
                break;
            }
            obs = state_t;
        }
        let episode_targets = env.agents.get(0).unwrap().targets_found;
        let episode_steps = total_steps - last_step;
        if (!renderer.render) {
            total_targets += episode_targets;
            total_rewards += episode_rewards;
        }
        let targets_per_step = match total_targets {
            0 => 0f64,
            x => x as f64 / total_steps as f64,
        };
        let targets_per_episode = match total_targets {
            0 => 0f64,
            x => x as f64 / episode as f64,
        };

        //println!("episode {}(steps {}, targets {}), total step {}, total targets {}, target/step {}, reward {}, epsilon {}", episode, total_step - last_step, env.agents.get(0).unwrap().collected_targets.len() - 1, total_step, found_targets, found_targets/total_step as f64, total_reward, agent.my_noise.epsilon);
        let tmp_target_step_avg_100 = target_step_avg_100.feed(episode_targets as f64/episode_steps as f64);
        let tmp_target_avg_100 = target_avg_100.feed(episode_targets as f64);

        max_target_avg_100 = max_target_avg_100.max(tmp_target_avg_100);

        let record = StatsRow {
            training_steps: episode * opts.TRAINING_BATCH_SIZE as i32,
            episode,
            episode_steps,
            episode_targets,
            episode_rewards,
            total_steps,
            total_targets,
            targets_per_step,
            targets_per_episode,
            rewards_per_step: total_rewards / total_steps as f64,
            target_step_avg_100: tmp_target_step_avg_100,
            target_avg_100: tmp_target_avg_100,
            max_target_avg_100: max_target_avg_100,
            epsilon: agent.my_noise.epsilon,
            actor_loss: agent.actor_loss,
            critic_loss: agent.critic_loss,
        };
        dbg!(&record);
        dbg!(total_steps as f64 / start.elapsed().as_secs_f64());
        wtr.serialize(record).unwrap();
        wtr.flush().unwrap();
        last_step = total_steps;
        for _ in 0..opts.TRAINING_ITERATIONS {
            agent.train(opts.TRAINING_BATCH_SIZE);
        }
    }
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
    #[clap(long, default_value = "0.999999")]
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
    training_steps: i32,
    episode: i32,
    episode_steps: i32,
    episode_targets: i32,
    episode_rewards: f64,
    total_steps: i32,
    total_targets: i32,
    targets_per_step: f64,
    targets_per_episode: f64,
    rewards_per_step: f64,
    target_step_avg_100: f64,
    target_avg_100: f64,
    max_target_avg_100: f64,
    actor_loss: f64,
    critic_loss: f64,
    epsilon: f64,
}
