use crate::condition::Condition;
use crate::config::Configuration;
use crate::effect::Effect;
use crate::mark::Mark;
use crate::perception::Perception;

#[derive(Clone, Debug)]
pub struct Classifier<const N: usize> {
    pub condition: Condition<N>,
    pub action: Option<usize>,
    pub effect: Effect<N>,
    pub mark: Mark<N>,
    pub q: f64,
    pub r: f64,
    pub ir: f64,
    pub num: u32,
    pub exp: u32,
    pub talp: Option<u64>,
    pub tga: u64,
    pub tav: f64,
    pub ee: bool,
}

impl<const N: usize> Classifier<N> {
    pub fn general(action: Option<usize>, config: &Configuration) -> Self {
        Self {
            condition: Condition::all_wildcard(),
            action,
            effect: Effect::all_wildcard(),
            mark: Mark::new(),
            q: config.initial_q,
            r: config.initial_r,
            ir: config.initial_ir,
            num: 1,
            exp: 1,
            talp: None,
            tga: 0,
            tav: 0.0,
            ee: false,
        }
    }

    pub fn covering(action: usize, time: u64, config: &Configuration) -> Self {
        Self {
            condition: Condition::all_wildcard(),
            action: Some(action),
            effect: Effect::all_wildcard(),
            mark: Mark::new(),
            q: config.initial_q,
            r: 0.0,
            ir: config.initial_ir,
            num: 1,
            exp: 0,
            talp: Some(time),
            tga: time,
            tav: 0.0,
            ee: false,
        }
    }

    pub fn copy_from(other: &Classifier<N>, time: u64) -> Self {
        Self {
            condition: other.condition,
            action: other.action,
            effect: other.effect,
            mark: Mark::new(),
            q: other.q,
            r: other.r,
            ir: other.ir,
            num: 1,
            exp: 1,
            talp: Some(time),
            tga: time,
            tav: other.tav,
            ee: false,
        }
    }

    pub fn fitness(&self) -> f64 {
        self.q * self.r
    }

    pub fn increase_quality(&mut self, beta: f64) {
        self.q += beta * (1.0 - self.q);
    }

    pub fn decrease_quality(&mut self, beta: f64) {
        self.q -= beta * self.q;
    }

    pub fn increase_experience(&mut self) {
        self.exp += 1;
    }

    pub fn update_application_average(&mut self, time: u64, beta: f64) {
        let previous = self.talp.unwrap_or(time);
        let last_applied = time as f64 - previous as f64 - self.tav;
        if (self.exp as f64) < 1.0 / beta {
            self.tav += last_applied / self.exp as f64;
        } else {
            self.tav += beta * last_applied;
        }
        self.talp = Some(time);
    }

    pub fn is_reliable(&self, theta_r: f64) -> bool {
        self.q > theta_r
    }

    pub fn is_inadequate(&self, theta_i: f64) -> bool {
        self.q < theta_i
    }

    pub fn is_more_general(&self, other: &Classifier<N>) -> bool {
        self.condition.specificity() < other.condition.specificity()
    }

    pub fn does_anticipate_correctly(&self, p0: &Perception<N>, p1: &Perception<N>) -> bool {
        self.effect.anticipates_correctly(p0, p1)
    }

    pub fn does_anticipate_change(&self) -> bool {
        self.effect.specify_change()
    }

    pub fn does_match(&self, perception: &Perception<N>) -> bool {
        self.condition.does_match(perception)
    }

    pub fn set_mark(&mut self, perception: &Perception<N>) {
        if self.mark.set_using_condition(&self.condition, perception) {
            self.ee = false;
        }
    }

    pub fn is_marked(&self) -> bool {
        self.mark.is_marked()
    }

    pub fn specialize(&mut self, p0: &Perception<N>, p1: &Perception<N>, leave_specialized: bool) {
        for index in 0..N {
            if leave_specialized && !self.effect.get(index).is_wildcard() {
                continue;
            }
            if p0.get(index) != p1.get(index) {
                if self.effect.get(index).is_wildcard() {
                    self.effect.set(index, p1.get(index));
                }
                self.condition.set(index, p0.get(index));
            }
        }
    }
}

impl<const N: usize> PartialEq for Classifier<N> {
    fn eq(&self, other: &Self) -> bool {
        self.condition == other.condition
            && self.action == other.action
            && self.effect == other.effect
    }
}
