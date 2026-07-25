#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AlpGenVariant {
    Pyalcs,
    Butz,
}

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
    pub alp_gen_variant: AlpGenVariant,
}

impl Configuration {
    pub fn default_protocol() -> Self {
        Self {
            number_of_possible_actions: 8,
            beta: 0.05,
            gamma: 0.95,
            theta_i: 0.1,
            theta_r: 0.9,
            theta_exp: 20,
            theta_as: 20,
            theta_ga: 100,
            mu: 0.3,
            chi: 0.8,
            u_max: 100_000,
            epsilon: 0.5,
            initial_q: 0.5,
            initial_r: 0.5,
            initial_ir: 0.0,
            do_ga: false,
            do_pee: false,
            do_action_planning: false,
            do_subsumption: true,
            alp_gen_variant: AlpGenVariant::Pyalcs,
        }
    }

    pub fn mpx() -> Self {
        Self {
            number_of_possible_actions: 2,
            ..Self::default_protocol()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpx_uses_two_actions_and_leaves_other_defaults() {
        let mpx = Configuration::mpx();
        assert_eq!(mpx.number_of_possible_actions, 2);
        assert!(!mpx.do_ga);
        assert_eq!(mpx.u_max, Configuration::default_protocol().u_max);
        assert_eq!(Configuration::default_protocol().number_of_possible_actions, 8);
    }
}
