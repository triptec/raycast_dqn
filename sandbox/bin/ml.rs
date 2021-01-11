use tch::{Tensor, nn, Device, Reduction};
use tch::kind::{FLOAT_CPU, INT64_CPU};
use tch::nn::OptimizerConfig;
use tch::kind::Kind::Float;

pub struct ReplayBuffer {
    obs: Tensor,
    next_obs: Tensor,
    rewards: Tensor,
    actions: Tensor,
    capacity: usize,
    len: usize,
    i: usize,
}

impl ReplayBuffer {
    pub(crate) fn new(capacity: usize, num_obs: usize, num_actions: usize) -> Self {
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

    pub(crate) fn push(&mut self, obs: &Tensor, actions: &Tensor, reward: &Tensor, next_obs: &Tensor) {
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

pub struct Actor {
    var_store: nn::VarStore,
    network: nn::Sequential,
    device: Device,
    num_obs: usize,
    num_actions: usize,
    optimizer: nn::Optimizer<nn::Adam>,
    learning_rate: f64,
    layers: Vec<i64>,
    gamma: f64,
    tau: f64,

}

impl Clone for Actor {
    fn clone(&self) -> Self {
        let mut new = Self::new(self.num_obs, self.num_actions, self.gamma, self.tau, self.learning_rate, self.layers.clone());
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl Actor {
    pub(crate) fn new(num_obs: usize, num_actions: usize, gamma: f64, tau: f64, learning_rate: f64, layers: Vec<i64>,) -> Self {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let optimizer = nn::Adam::default()
            .build(&var_store, learning_rate)
            .unwrap();
        let path = &var_store.root();
        let mut network = nn::seq();
        for index in 0..layers.len() {
            if index == 0 {
                network = network
                    .add(nn::linear(path / "first", num_obs as _, layers.get(index).unwrap().clone(), Default::default()))
                    .add_fn(|xs| xs.relu());
            } else {
                network = network
                    .add(nn::linear(path / format!("l{}", index), layers.get(index).unwrap().clone(), layers.get(index).unwrap().clone(), Default::default()))
                    .add_fn(|xs| xs.relu());
            }

            if index == layers.len() - 1 {
                network = network
                    .add(nn::linear(
                        path / "last",
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
            device: path.device(),
            num_obs,
            num_actions,
            var_store,
            optimizer,
            learning_rate,
            layers,
            gamma,
            tau,
        }
    }
    pub fn forward(&self, obs: &Tensor) -> Tensor {
        obs.to_device(self.device).apply(&self.network)
    }
}

pub struct Model_a2c {
    pub actor: Actor,
    actor_target: Actor,
    gamma: f64,
    tau: f64,
    num_obs: usize,
    num_actions: usize,
    learning_rate: f64,
    layers: Vec<i64>,
}

impl Model_a2c {
    pub fn new(
        num_obs: usize, num_actions: usize,
        gamma: f64,
        tau: f64,
        learning_rate: f64, layers: Vec<i64>,
    ) -> Self {
        let actor = Actor::new(num_obs, num_actions, gamma, tau, learning_rate, layers.clone());
        let actor_target = actor.clone();
        Self {
            actor,
            actor_target,
            gamma,
            tau,
            num_obs,
            num_actions,
            learning_rate,
            layers: layers.clone(),
        }
    }

    pub fn train(&mut self, replay_buffer: &mut ReplayBuffer, batch_size: usize) {
        let (states, actions, rewards, next_states) =
            match replay_buffer.random_batch(batch_size) {
                Some(v) => v,
                _ => return, // We don't have enough samples for training yet.
            };

        let mut q_target = rewards.copy() + (self.gamma * self.actor_target.forward(&next_states)).detach();
        let q = self.actor.forward(&states);

        let diff = q_target.copy() - q.copy();
        let actions_taken = actions.argmax(-1, false);
        let mut q1 = q.copy();
        for i in 0..batch_size {
            let mut row = q1.get(i as i64);
            let mut t = row.get(i64::from(actions_taken.get(i as i64)));

            t.copy_(&q_target.get(i as i64).get(i64::from(actions_taken.get(i as i64))));
            //dbg!(&q.get(i as i64), &q1.get(i as i64));
        }

        let x = diff.argmax(-2, false);
        for i in 0..5 {
            let t = &x.get(i);
            let index = i64::from(t);
            for _ in 0..4 {
                replay_buffer.push(&states.get(index).copy(), &actions.get(index).copy(), &rewards.get(index).copy(), &next_states.get(index).copy());
            }
        }

        let value_loss = q.smooth_l1_loss(&q1, Reduction::Mean, 1.0);// * self.tau;
        //let value_loss = q.smooth_l1_loss(&q_target, Reduction::Mean, 1.0);
        //dbg!(&q_target.get(index), &q.get(index), &rewards.get(index), &value_loss);

        //let value_loss = (&diff * &diff).mean(Float);
        self.actor.optimizer.zero_grad();
        value_loss.backward();
        self.actor.optimizer.step();

        Self::track(
            &mut self.actor_target.var_store,
            &self.actor.var_store,
            self.tau,
        );
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
}