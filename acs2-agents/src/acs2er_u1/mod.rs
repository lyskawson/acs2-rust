use acs2_core::agents::acs2er::ReplayConfiguration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiagnosticConfiguration {
    pub capacity: usize,
    pub warmup: usize,
}

impl DiagnosticConfiguration {
    pub fn replay(self) -> ReplayConfiguration {
        ReplayConfiguration {
            buffer_size: self.capacity,
            min_samples: self.warmup,
            samples_number: 1,
        }
    }
}
