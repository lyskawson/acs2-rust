#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReplaySchedule {
    #[default]
    Constant,
    ThreeToOne {
        switch_steps: u64,
    },
}

impl ReplaySchedule {
    pub fn rate(self, episode_end: u64, constant: usize) -> usize {
        match self {
            Self::Constant => constant,
            Self::ThreeToOne { switch_steps } => {
                if episode_end <= switch_steps {
                    3
                } else {
                    1
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReplayOrder {
    #[default]
    Independent,
    Backward,
    RandomPass,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayPlan {
    pub schedule: ReplaySchedule,
    pub order: ReplayOrder,
}

impl ReplayPlan {
    pub fn validate(self) {
        if let ReplaySchedule::ThreeToOne { switch_steps } = self.schedule {
            assert!(switch_steps > 0);
            assert_eq!(
                self.order,
                ReplayOrder::Independent,
                "schedule and pass order are separate axes"
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassCounts {
    pub passes: u64,
    pub requested_relabel: u64,
    pub fallbacks: u64,
    pub cut_passes: u64,
    pub available_steps: u64,
    pub emitted_steps: u64,
}

pub const ORDER_STREAM: u64 = 9;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScheduleCounts {
    pub early_steps: u64,
    pub late_steps: u64,
    pub first_late_episode_end: Option<u64>,
}
