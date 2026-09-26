use acs2_core::goal::GoalLayout;
use acs2_core::knowledge::Transition;
use acs2_core::rng::ChaChaRandomSource;
use acs2_core::symbol::Symbol;
use acs2_envs::goal::bit_flipping::BitFlipping;
use acs2_envs::goal::hand_eye::HandEye3;
use acs2_envs::goal::knowledge::with_wildcard_goals;
use acs2_envs::goal::taxi::Taxi;

fn compare<const S: usize, const G: usize, const M: usize>(transitions: Vec<Transition<S>>) {
    let joined: Vec<_> = with_wildcard_goals::<S, G, M>(
        transitions
            .iter()
            .map(|t| Transition::new(t.p0, t.action, t.p1)),
    )
    .collect();
    assert_eq!(joined.len(), transitions.len());
    for (full, plain) in joined.iter().zip(transitions) {
        let (p0, g0) = GoalLayout::<S, G, M>::split(&full.p0);
        let (p1, g1) = GoalLayout::<S, G, M>::split(&full.p1);
        assert_eq!(p0, plain.p0);
        assert_eq!(p1, plain.p1);
        assert_eq!(full.action, plain.action);
        assert_eq!(g0.symbols, [Symbol::Wildcard; G]);
        assert_eq!(g1.symbols, [Symbol::Wildcard; G]);
    }
}

#[test]
fn each_non_maze_knowledge_set_can_be_lifted_without_exposing_a_real_goal() {
    let rng = || Box::new(ChaChaRandomSource::from_seed(0));
    compare::<4, 4, 8>(
        BitFlipping::<4>::new(rng())
            .knowledge_transitions()
            .collect(),
    );
    compare::<10, 2, 12>(HandEye3::new(3, rng()).knowledge_transitions());
    compare::<3, 1, 4>(Taxi::new(3, rng()).knowledge_transitions());
}
