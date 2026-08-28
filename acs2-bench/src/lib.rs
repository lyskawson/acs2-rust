use acs2_core::acs2er::ReplayConfiguration;
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AgentChoice {
    Acs2,
    Acs2Er,
}

pub fn parse_agent(value: &str) -> AgentChoice {
    match value {
        "acs2" => AgentChoice::Acs2,
        "acs2er" => AgentChoice::Acs2Er,
        other => panic!("unknown agent {other} (expected acs2 or acs2er)"),
    }
}

pub fn agent_label(agent: AgentChoice) -> &'static str {
    match agent {
        AgentChoice::Acs2 => "acs2",
        AgentChoice::Acs2Er => "acs2er",
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AgentOptions {
    pub agent: AgentChoice,
    pub replay: ReplayConfiguration,
}

impl AgentOptions {
    pub fn try_parse_flag<I>(&mut self, flag: &str, args: &mut I) -> bool
    where
        I: Iterator<Item = String>,
    {
        match flag {
            "--agent" => {
                self.agent = parse_agent(&args.next().expect("--agent needs acs2|acs2er"));
                true
            }
            "--er-buffer-size" => {
                self.replay.buffer_size = parse_size(args.next(), flag);
                true
            }
            "--er-min-samples" => {
                self.replay.min_samples = parse_size(args.next(), flag);
                true
            }
            "--er-samples-number" => {
                self.replay.samples_number = parse_size(args.next(), flag);
                true
            }
            _ => false,
        }
    }

    pub fn describe(&self) -> String {
        match self.agent {
            AgentChoice::Acs2 => "agent=acs2".to_string(),
            AgentChoice::Acs2Er => format!(
                "agent=acs2er er_buffer_size={} er_min_samples={} er_samples_number={}",
                self.replay.buffer_size, self.replay.min_samples, self.replay.samples_number,
            ),
        }
    }
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            agent: AgentChoice::Acs2,
            replay: ReplayConfiguration::default_protocol(),
        }
    }
}

fn parse_size(value: Option<String>, flag: &str) -> usize {
    value
        .unwrap_or_else(|| panic!("{flag} needs a value"))
        .parse()
        .unwrap_or_else(|_| panic!("{flag} must be a non-negative integer"))
}
