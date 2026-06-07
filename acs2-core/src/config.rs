#[derive(Clone, Debug)]
pub struct Configuration {
    pub number_of_possible_actions: usize,
    pub beta: f64,
    pub gamma: f64,
    pub theta_i: f64,
    pub theta_r: f64,
    pub theta_exp: u32,
    pub theta_as: u32,
    pub theta_ga: u32,
    pub mu: f64,
    pub chi: f64,
    pub u_max: u32,
    pub epsilon: f64,
    pub initial_q: f64,
    pub initial_r: f64,
    pub initial_ir: f64,
    pub do_ga: bool,
    pub do_pee: bool,
    pub do_action_planning: bool,
    pub do_subsumption: bool,
}

impl Configuration {
    pub fn default_protocol() -> Self {
        todo!()
    }
}
