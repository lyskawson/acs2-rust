use std::cell::Cell;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct MatchCounters {
    pub formations: u64,
    pub classifier_tests: u64,
}

thread_local! {
    static COUNTERS: Cell<Option<MatchCounters>> = const { Cell::new(None) };
}

pub fn start_match_counting() {
    COUNTERS.with(|counters| counters.set(Some(MatchCounters::default())));
}

pub fn read_match_counters() -> Option<MatchCounters> {
    COUNTERS.with(Cell::get)
}

pub fn stop_match_counting() -> Option<MatchCounters> {
    COUNTERS.with(|counters| counters.replace(None))
}

pub(crate) fn record_match_formation(population_size: usize) {
    COUNTERS.with(|counters| {
        if let Some(mut value) = counters.get() {
            value.formations += 1;
            value.classifier_tests += population_size as u64;
            counters.set(Some(value));
        }
    });
}
