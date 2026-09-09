use std::collections::BTreeSet;

use acs2_core::alp::expected_case;
use acs2_core::classifier::Classifier;
use acs2_core::condition::Condition;
use acs2_core::config::{AlpGenVariant, Configuration};
use acs2_core::effect::Effect;
use acs2_core::mark::Mark;
use acs2_core::perception::Perception;
use acs2_core::rng::ChaChaRandomSource;
use acs2_core::symbol::Symbol;

const W: Symbol = Symbol::Wildcard;

fn token(value: u8) -> Symbol {
    Symbol::Token(value)
}

fn over_specialized_classifier() -> Classifier<4> {
    let mut mark = Mark::<4>::new();
    let mut fourth: BTreeSet<Symbol> = BTreeSet::new();
    fourth.insert(token(0));
    mark.attributes[3] = fourth;

    Classifier {
        condition: Condition {
            symbols: [token(1), token(1), token(1), W],
        },
        action: Some(0),
        effect: Effect {
            symbols: [W, W, W, W],
        },
        mark,
        q: 0.9,
        r: 0.5,
        ir: 0.0,
        num: 1,
        exp: 5,
        talp: Some(0),
        tga: 0,
        tav: 0.0,
        ee: false,
    }
}

fn config_with(variant: AlpGenVariant, u_max: u32) -> Configuration {
    let mut config = Configuration::mpx();
    config.alp_gen_variant = variant;
    config.u_max = u_max;
    config
}

#[test]
fn pyalcs_variant_generalizes_parent_and_leaves_child_specialized() {
    let mut parent = over_specialized_classifier();
    let p0 = Perception::new([token(1), token(1), token(1), token(1)]);
    let config = config_with(AlpGenVariant::Pyalcs, 1);
    let mut rng = ChaChaRandomSource::from_seed(7);

    assert_eq!(parent.specified_unchanging_attributes().len(), 3);

    let child = expected_case(&mut parent, &p0, 10, &config, &mut rng)
        .expect("difference is specific, so a child must be produced");

    assert_eq!(
        parent.condition.specificity(),
        0,
        "pyalcs variant must generalize the parent in place down to u_max-1 = 0 unchanging"
    );
    assert!(
        child.condition.specificity() >= 3,
        "pyalcs child must remain fully specialized (never generalized by the branch)"
    );
}

#[test]
fn butz_variant_generalizes_child_and_leaves_parent_untouched() {
    let mut parent = over_specialized_classifier();
    let p0 = Perception::new([token(1), token(1), token(1), token(1)]);
    let config = config_with(AlpGenVariant::Butz, 1);
    let mut rng = ChaChaRandomSource::from_seed(7);

    let child = expected_case(&mut parent, &p0, 10, &config, &mut rng)
        .expect("difference is specific, so a child must be produced");

    assert_eq!(
        parent.condition.specificity(),
        3,
        "butz variant must not mutate the parent condition"
    );
    assert!(
        child.condition.specificity() <= 1,
        "butz child must be generalized so spec + spec_new does not exceed u_max = 1"
    );
}

#[test]
fn disabled_u_max_leaves_both_parent_and_child_untouched_by_generalization() {
    for variant in [AlpGenVariant::Pyalcs, AlpGenVariant::Butz] {
        let mut parent = over_specialized_classifier();
        let p0 = Perception::new([token(1), token(1), token(1), token(1)]);
        let config = config_with(variant, 100_000);
        let mut rng = ChaChaRandomSource::from_seed(7);

        let child = expected_case(&mut parent, &p0, 10, &config, &mut rng)
            .expect("difference is specific, so a child must be produced");

        assert_eq!(
            parent.condition.specificity(),
            3,
            "disabled u_max must leave the parent specialize-only"
        );
        assert_eq!(
            child.condition.specificity(),
            4,
            "disabled u_max: child is the parent condition specialized with the one-position diff"
        );
    }
}
