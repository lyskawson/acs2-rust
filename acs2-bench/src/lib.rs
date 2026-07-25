use acs2_core::config::AlpGenVariant;
use acs2_envs::multiplexer::control_bits_for;

#[derive(Clone, Copy)]
pub enum UMaxMode {
    Default,
    Fixed(u32),
    Derived,
}

pub fn parse_variant(value: &str) -> AlpGenVariant {
    match value {
        "pyalcs" => AlpGenVariant::Pyalcs,
        "butz" => AlpGenVariant::Butz,
        other => panic!("unknown alp-gen-variant {other}"),
    }
}

pub fn variant_label(variant: AlpGenVariant) -> &'static str {
    match variant {
        AlpGenVariant::Pyalcs => "pyalcs",
        AlpGenVariant::Butz => "butz",
    }
}

pub fn derived_u_max(size: usize, variant: AlpGenVariant) -> u32 {
    let address_bits = control_bits_for(size + 1);
    let offset = match variant {
        AlpGenVariant::Pyalcs => 2,
        AlpGenVariant::Butz => 3,
    };
    (address_bits + offset) as u32
}

pub fn resolve_u_max(mode: UMaxMode, default: u32, size: usize, variant: AlpGenVariant) -> u32 {
    match mode {
        UMaxMode::Default => default,
        UMaxMode::Fixed(value) => value,
        UMaxMode::Derived => derived_u_max(size, variant),
    }
}

pub fn parse_u_max_mode(value: &str) -> UMaxMode {
    if value == "derived" {
        UMaxMode::Derived
    } else {
        UMaxMode::Fixed(value.parse().expect("--u-max must be an integer or 'derived'"))
    }
}
