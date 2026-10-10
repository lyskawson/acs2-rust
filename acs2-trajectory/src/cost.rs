#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObjectiveCost {
    pub reward_evaluations: u64,
    pub reach_evaluations: u64,
}

impl ObjectiveCost {
    pub fn reach(evaluations: u64) -> Self {
        Self {
            reward_evaluations: 0,
            reach_evaluations: evaluations,
        }
    }
    pub fn total(self) -> u64 {
        self.reward_evaluations + self.reach_evaluations
    }
    pub fn add(&mut self, other: Self) {
        self.reward_evaluations += other.reward_evaluations;
        self.reach_evaluations += other.reach_evaluations;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CostByPurpose {
    pub scoring: ObjectiveCost,
    pub selection: ObjectiveCost,
    pub provenance: ObjectiveCost,
}

impl CostByPurpose {
    pub fn total(self) -> ObjectiveCost {
        let mut total = self.scoring;
        total.add(self.selection);
        total.add(self.provenance);
        total
    }
    pub fn add(&mut self, other: Self) {
        self.scoring.add(other.scoring);
        self.selection.add(other.selection);
        self.provenance.add(other.provenance);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FactQueries {
    pub current_state: u64,
    pub earlier_states: u64,
}

impl FactQueries {
    pub fn total(self) -> u64 {
        self.current_state + self.earlier_states
    }
    pub fn as_provenance(self) -> CostByPurpose {
        CostByPurpose {
            provenance: ObjectiveCost::reach(self.total()),
            ..CostByPurpose::default()
        }
    }
}
