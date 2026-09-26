use acs2_core::goal::{Goal, GoalLayout};
use acs2_core::knowledge::Transition;
use acs2_core::symbol::Symbol;

pub fn with_wildcard_goals<const S: usize, const G: usize, const M: usize>(
    transitions: impl IntoIterator<Item = Transition<S>>,
) -> impl Iterator<Item = Transition<M>> {
    let wildcard = Goal::new([Symbol::Wildcard; G]);
    transitions.into_iter().map(move |transition| {
        Transition::new(
            GoalLayout::<S, G, M>::join(&transition.p0, &wildcard),
            transition.action,
            GoalLayout::<S, G, M>::join(&transition.p1, &wildcard),
        )
    })
}
