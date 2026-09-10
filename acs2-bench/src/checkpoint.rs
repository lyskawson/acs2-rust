use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use acs2_core::acs2er::ReplaySample;
use acs2_core::checkpoint::AgentState;
use acs2_core::classifier::Classifier;
use acs2_core::condition::Condition;
use acs2_core::effect::Effect;
use acs2_core::mark::Mark;
use acs2_core::perception::Perception;
use acs2_core::rng::RngState;
use acs2_core::symbol::Symbol;

// Version 2 requires the trailing terminator, so a version-1 file is refused
// rather than read as a complete one that happens to end early.
pub const FORMAT: &str = "acs2-checkpoint 2";
const TERMINATOR: &str = "end";

const WILDCARD: &str = "--";
const UNSET: &str = "-";

#[derive(Clone, Debug)]
pub struct CheckpointSettings {
    pub path: PathBuf,
    pub every: u64,
    pub allow_eval_interval_change: bool,
}

#[derive(Clone, Debug)]
pub struct RunState<const N: usize> {
    pub identity: String,
    pub eval_interval: u64,
    pub trials_used: u64,
    pub time: u64,
    pub trials_since_eval: u64,
    pub peak_macro_population: usize,
    pub peak_rss_bytes: u64,
    pub wall_seconds: f64,
    pub final_knowledge: f64,
    pub knowledge_trials: Option<u64>,
    pub verdict: Option<String>,
    pub env_rng: RngState,
    pub agent: AgentState<N>,
}

fn encode_float(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn decode_float(text: &str) -> f64 {
    f64::from_bits(
        u64::from_str_radix(text, 16).unwrap_or_else(|_| panic!("checkpoint float {text}")),
    )
}

fn encode_symbol(symbol: Symbol, into: &mut String) {
    match symbol {
        Symbol::Wildcard => into.push_str(WILDCARD),
        Symbol::Token(value) => {
            let _ = write!(into, "{value:02x}");
        }
    }
}

fn decode_symbol(text: &str) -> Symbol {
    if text == WILDCARD {
        Symbol::Wildcard
    } else {
        Symbol::Token(
            u8::from_str_radix(text, 16).unwrap_or_else(|_| panic!("checkpoint symbol {text}")),
        )
    }
}

fn encode_symbols(symbols: &[Symbol]) -> String {
    let mut text = String::with_capacity(symbols.len() * 2);
    for &symbol in symbols {
        encode_symbol(symbol, &mut text);
    }
    text
}

fn decode_symbols<const N: usize>(text: &str) -> [Symbol; N] {
    let bytes = text.as_bytes();
    assert_eq!(
        bytes.len(),
        N * 2,
        "checkpoint holds a {}-symbol string where {N} were expected",
        bytes.len() / 2
    );
    core::array::from_fn(|index| decode_symbol(&text[index * 2..index * 2 + 2]))
}

fn encode_mark<const N: usize>(mark: &Mark<N>) -> String {
    let mut groups: Vec<String> = Vec::new();
    for (index, attribute) in mark.attributes.iter().enumerate() {
        if attribute.is_empty() {
            continue;
        }
        let mut text = String::new();
        for &symbol in attribute.iter() {
            if !text.is_empty() {
                text.push(',');
            }
            encode_symbol(symbol, &mut text);
        }
        groups.push(format!("{index}:{text}"));
    }
    if groups.is_empty() {
        UNSET.to_string()
    } else {
        groups.join("|")
    }
}

fn decode_mark<const N: usize>(text: &str) -> Mark<N> {
    let mut mark = Mark::new();
    if text == UNSET {
        return mark;
    }
    for group in text.split('|') {
        let (index, symbols) = group
            .split_once(':')
            .unwrap_or_else(|| panic!("checkpoint mark group {group}"));
        let index: usize = index
            .parse()
            .unwrap_or_else(|_| panic!("checkpoint mark index {index}"));
        let attribute: BTreeSet<Symbol> = symbols.split(',').map(decode_symbol).collect();
        mark.attributes[index] = attribute;
    }
    mark
}

fn encode_rng(state: &RngState) -> String {
    let mut seed = String::with_capacity(64);
    for byte in state.seed {
        let _ = write!(seed, "{byte:02x}");
    }
    format!("{seed} {} {}", state.stream, state.word_pos)
}

fn decode_rng(fields: &[&str]) -> RngState {
    assert_eq!(fields.len(), 3, "checkpoint rng line takes three fields");
    let seed_text = fields[0];
    assert_eq!(seed_text.len(), 64, "checkpoint rng seed is 32 bytes");
    let seed = core::array::from_fn(|index| {
        u8::from_str_radix(&seed_text[index * 2..index * 2 + 2], 16)
            .unwrap_or_else(|_| panic!("checkpoint rng seed {seed_text}"))
    });
    RngState {
        seed,
        stream: fields[1].parse().expect("checkpoint rng stream"),
        word_pos: fields[2].parse().expect("checkpoint rng word position"),
    }
}

fn encode_classifier<const N: usize>(classifier: &Classifier<N>) -> String {
    format!(
        "c {} {} {} {} {} {} {} {} {} {} {} {} {}",
        encode_symbols(&classifier.condition.symbols),
        classifier
            .action
            .map(|action| action.to_string())
            .unwrap_or_else(|| UNSET.to_string()),
        encode_symbols(&classifier.effect.symbols),
        encode_mark(&classifier.mark),
        encode_float(classifier.q),
        encode_float(classifier.r),
        encode_float(classifier.ir),
        classifier.num,
        classifier.exp,
        classifier
            .talp
            .map(|talp| talp.to_string())
            .unwrap_or_else(|| UNSET.to_string()),
        classifier.tga,
        encode_float(classifier.tav),
        u8::from(classifier.ee),
    )
}

fn decode_classifier<const N: usize>(fields: &[&str]) -> Classifier<N> {
    assert_eq!(fields.len(), 13, "checkpoint classifier line takes 13 fields");
    Classifier {
        condition: Condition {
            symbols: decode_symbols::<N>(fields[0]),
        },
        action: parse_optional(fields[1], "classifier action"),
        effect: Effect {
            symbols: decode_symbols::<N>(fields[2]),
        },
        mark: decode_mark::<N>(fields[3]),
        q: decode_float(fields[4]),
        r: decode_float(fields[5]),
        ir: decode_float(fields[6]),
        num: fields[7].parse().expect("checkpoint classifier numerosity"),
        exp: fields[8].parse().expect("checkpoint classifier experience"),
        talp: parse_optional(fields[9], "classifier talp"),
        tga: fields[10].parse().expect("checkpoint classifier tga"),
        tav: decode_float(fields[11]),
        ee: decode_flag(fields[12], "classifier ee"),
    }
}

fn encode_sample<const N: usize>(sample: &ReplaySample<N>) -> String {
    format!(
        "s {} {} {} {} {}",
        encode_symbols(&sample.state.symbols),
        sample.action,
        encode_float(sample.reward),
        encode_symbols(&sample.next_state.symbols),
        u8::from(sample.done),
    )
}

fn decode_sample<const N: usize>(fields: &[&str]) -> ReplaySample<N> {
    assert_eq!(fields.len(), 5, "checkpoint replay line takes five fields");
    ReplaySample {
        state: Perception::new(decode_symbols::<N>(fields[0])),
        action: fields[1].parse().expect("checkpoint replay action"),
        reward: decode_float(fields[2]),
        next_state: Perception::new(decode_symbols::<N>(fields[3])),
        done: decode_flag(fields[4], "replay done flag"),
    }
}

/// Booleans are decoded strictly.
///
/// `text == "1"` would read every other string as `false`, and a file truncated exactly
/// after the space before a trailing flag still yields a field -- an empty one. That
/// turns a terminal replay sample into a bootstrapped one and silently changes what the
/// agent learns from it.
fn decode_flag(text: &str, what: &str) -> bool {
    match text {
        "0" => false,
        "1" => true,
        other => panic!("checkpoint {what} is {other:?}, expected 0 or 1"),
    }
}

fn parse_optional<T: std::str::FromStr>(text: &str, what: &str) -> Option<T> {
    if text == UNSET {
        None
    } else {
        Some(
            text.parse()
                .unwrap_or_else(|_| panic!("checkpoint {what} {text}")),
        )
    }
}

pub fn render<const N: usize>(state: &RunState<N>) -> String {
    let mut text = String::new();
    let _ = writeln!(text, "{FORMAT}");
    let _ = writeln!(text, "identity {}", state.identity);
    let _ = writeln!(
        text,
        "run trials={} time={} since_eval={} peak_pop={} peak_rss={} wall={} knowledge={} knowledge_trials={} eval_interval={} verdict={}",
        state.trials_used,
        state.time,
        state.trials_since_eval,
        state.peak_macro_population,
        state.peak_rss_bytes,
        encode_float(state.wall_seconds),
        encode_float(state.final_knowledge),
        state
            .knowledge_trials
            .map(|trials| trials.to_string())
            .unwrap_or_else(|| UNSET.to_string()),
        state.eval_interval,
        state.verdict.clone().unwrap_or_else(|| UNSET.to_string()),
    );
    let _ = writeln!(text, "env_rng {}", encode_rng(&state.env_rng));
    let _ = writeln!(text, "agent_rng {}", encode_rng(&state.agent.rng));
    let _ = writeln!(text, "population {}", state.agent.population.len());
    for classifier in &state.agent.population {
        let _ = writeln!(text, "{}", encode_classifier(classifier));
    }
    match &state.agent.replay {
        Some(samples) => {
            let _ = writeln!(text, "replay {}", samples.len());
            for sample in samples {
                let _ = writeln!(text, "{}", encode_sample(sample));
            }
        }
        None => {
            let _ = writeln!(text, "replay {UNSET}");
        }
    }
    let _ = writeln!(text, "{TERMINATOR}");
    text
}

pub fn parse<const N: usize>(text: &str) -> RunState<N> {
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some(FORMAT),
        "unrecognised checkpoint format header"
    );

    let identity = lines
        .next()
        .and_then(|line| line.strip_prefix("identity "))
        .expect("checkpoint carries no identity line")
        .to_string();

    let run = lines.next().expect("checkpoint carries no run line");
    let fields = keyed_fields(run, "run");
    let verdict = match field(&fields, "verdict") {
        value if value == UNSET => None,
        value => Some(value.to_string()),
    };

    let env_rng = decode_rng(&tail(lines.next().expect("checkpoint env_rng"), "env_rng"));
    let agent_rng = decode_rng(&tail(
        lines.next().expect("checkpoint agent_rng"),
        "agent_rng",
    ));

    let population_size: usize = tail(
        lines.next().expect("checkpoint population"),
        "population",
    )[0]
        .parse()
        .expect("checkpoint population size");
    let mut population = Vec::with_capacity(population_size);
    for _ in 0..population_size {
        let line = lines.next().expect("checkpoint truncated in the population");
        population.push(decode_classifier::<N>(&tail(line, "c")));
    }

    let replay_header = tail(lines.next().expect("checkpoint replay"), "replay");
    let replay = if replay_header[0] == UNSET {
        None
    } else {
        let count: usize = replay_header[0].parse().expect("checkpoint replay size");
        let mut samples = Vec::with_capacity(count);
        for _ in 0..count {
            let line = lines.next().expect("checkpoint truncated in the replay buffer");
            samples.push(decode_sample::<N>(&tail(line, "s")));
        }
        Some(samples)
    };

    assert_eq!(
        lines.next(),
        Some(TERMINATOR),
        "the checkpoint is truncated or carries trailing records"
    );
    assert_eq!(lines.next(), None, "the checkpoint carries trailing records");

    RunState {
        identity,
        eval_interval: field(&fields, "eval_interval").parse().expect("eval_interval"),
        trials_used: field(&fields, "trials").parse().expect("trials"),
        time: field(&fields, "time").parse().expect("time"),
        trials_since_eval: field(&fields, "since_eval").parse().expect("since_eval"),
        peak_macro_population: field(&fields, "peak_pop").parse().expect("peak_pop"),
        peak_rss_bytes: field(&fields, "peak_rss").parse().expect("peak_rss"),
        wall_seconds: decode_float(field(&fields, "wall")),
        final_knowledge: decode_float(field(&fields, "knowledge")),
        knowledge_trials: parse_optional(field(&fields, "knowledge_trials"), "knowledge_trials"),
        verdict,
        env_rng,
        agent: AgentState {
            population,
            rng: agent_rng,
            replay,
        },
    }
}

fn tail<'a>(line: &'a str, keyword: &str) -> Vec<&'a str> {
    let mut fields = line.split(' ');
    assert_eq!(
        fields.next(),
        Some(keyword),
        "checkpoint line does not start with {keyword}"
    );
    fields.collect()
}

fn keyed_fields<'a>(line: &'a str, keyword: &str) -> Vec<(&'a str, &'a str)> {
    tail(line, keyword)
        .into_iter()
        .map(|field| {
            field
                .split_once('=')
                .unwrap_or_else(|| panic!("checkpoint {keyword} field {field}"))
        })
        .collect()
}

fn field<'a>(fields: &[(&'a str, &'a str)], key: &str) -> &'a str {
    fields
        .iter()
        .find(|(name, _)| *name == key)
        .unwrap_or_else(|| panic!("checkpoint run line has no {key}"))
        .1
}

/// Staging file for the write-then-rename publish.
///
/// It appends to the whole destination file name rather than replacing an extension,
/// for two reasons a `with_extension("partial")` version got wrong: a destination
/// already ending in `.partial` would stage onto itself and truncate the only
/// recoverable copy, and `run.ckpt` and `run.backup` would stage onto one shared file.
/// The process id keeps two writers off each other's staging file; it does not make
/// concurrent writers safe, which is what job dependencies are for.
fn staging_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .unwrap_or_else(|| panic!("checkpoint path {} names no file", path.display()))
        .to_os_string();
    name.push(format!(".partial.{}", std::process::id()));
    path.with_file_name(name)
}

pub fn write<const N: usize>(path: &Path, state: &RunState<N>) {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).expect("cannot create the checkpoint directory");
        }
    }
    let staging = staging_path(path);
    fs::write(&staging, render(state)).expect("cannot write the checkpoint");
    fs::rename(&staging, path).expect("cannot publish the checkpoint");
}

pub fn read<const N: usize>(path: &Path) -> RunState<N> {
    parse::<N>(&fs::read_to_string(path).expect("cannot read the checkpoint"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use acs2_core::config::Configuration;

    fn state() -> RunState<7> {
        let mut classifier = Classifier::<7>::general(Some(1), &Configuration::mpx());
        classifier.condition.set(2, Symbol::Token(b'0'));
        classifier.effect.set(6, Symbol::Token(b'1'));
        classifier.mark.attributes[3].insert(Symbol::Token(b'1'));
        classifier.mark.attributes[3].insert(Symbol::Wildcard);
        classifier.mark.attributes[5].insert(Symbol::Token(b'0'));
        classifier.q = 0.123_456_789_012_345_67;
        classifier.tav = 1.0 / 3.0;
        classifier.talp = Some(9);
        classifier.ee = true;

        let unmarked = Classifier::<7>::general(None, &Configuration::mpx());

        RunState {
            identity: "size=6 seed=42".to_string(),
            eval_interval: 750,
            trials_used: 1500,
            time: 1500,
            trials_since_eval: 500,
            peak_macro_population: 41,
            peak_rss_bytes: 12_345,
            wall_seconds: 1.0 / 7.0,
            final_knowledge: 0.749_9,
            knowledge_trials: Some(1000),
            verdict: Some("TRIALS-LIMITED".to_string()),
            env_rng: RngState {
                seed: [7; 32],
                stream: 3,
                word_pos: 1_234_567_890_123,
            },
            agent: AgentState {
                population: vec![classifier, unmarked],
                rng: RngState {
                    seed: [9; 32],
                    stream: 0,
                    word_pos: 64,
                },
                replay: None,
            },
        }
    }

    #[test]
    fn a_rendered_checkpoint_parses_back_to_the_same_state() {
        let original = state();
        let restored = parse::<7>(&render(&original));

        assert_eq!(restored.identity, original.identity);
        assert_eq!(restored.trials_used, original.trials_used);
        assert_eq!(restored.time, original.time);
        assert_eq!(restored.trials_since_eval, original.trials_since_eval);
        assert_eq!(restored.peak_macro_population, original.peak_macro_population);
        assert_eq!(restored.peak_rss_bytes, original.peak_rss_bytes);
        assert_eq!(restored.wall_seconds.to_bits(), original.wall_seconds.to_bits());
        assert_eq!(
            restored.final_knowledge.to_bits(),
            original.final_knowledge.to_bits()
        );
        assert_eq!(restored.knowledge_trials, original.knowledge_trials);
        assert_eq!(restored.verdict, original.verdict);
        assert_eq!(restored.eval_interval, original.eval_interval);
        assert_eq!(restored.env_rng, original.env_rng);
        assert_eq!(restored.agent.rng, original.agent.rng);
        assert!(restored.agent.replay.is_none());
        assert_eq!(restored.agent.population.len(), 2);
        assert_eq!(render(&restored), render(&original));
    }

    #[test]
    fn every_classifier_field_survives_the_round_trip() {
        let original = state();
        let restored = parse::<7>(&render(&original));
        let before = &original.agent.population[0];
        let after = &restored.agent.population[0];

        assert_eq!(after.condition, before.condition);
        assert_eq!(after.effect, before.effect);
        assert_eq!(after.mark, before.mark);
        assert_eq!(after.action, before.action);
        assert_eq!(after.q.to_bits(), before.q.to_bits());
        assert_eq!(after.r.to_bits(), before.r.to_bits());
        assert_eq!(after.ir.to_bits(), before.ir.to_bits());
        assert_eq!(after.tav.to_bits(), before.tav.to_bits());
        assert_eq!(after.num, before.num);
        assert_eq!(after.exp, before.exp);
        assert_eq!(after.talp, before.talp);
        assert_eq!(after.tga, before.tga);
        assert_eq!(after.ee, before.ee);
        assert_eq!(restored.agent.population[1].action, None);
        assert!(!restored.agent.population[1].is_marked());
    }

    #[test]
    fn a_replay_buffer_survives_the_round_trip() {
        let mut original = state();
        original.agent.replay = Some(vec![ReplaySample {
            state: Perception::new([Symbol::Token(b'1'); 7]),
            action: 1,
            reward: 1000.0,
            next_state: Perception::new([Symbol::Token(b'0'); 7]),
            done: true,
        }]);
        let restored = parse::<7>(&render(&original));
        let samples = restored.agent.replay.expect("replay buffer");
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0], original.agent.replay.unwrap()[0]);
    }

    #[test]
    fn writing_a_checkpoint_leaves_no_partial_file_behind() {
        let directory = std::env::temp_dir().join(format!(
            "acs2-checkpoint-codec-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = directory.join("run.ckpt");
        write::<7>(&path, &state());
        assert!(path.exists());
        assert!(
            !staging_path(&path).exists(),
            "the staging file the writer actually uses must be gone"
        );
        assert_eq!(
            std::fs::read_dir(&directory).unwrap().count(),
            1,
            "nothing but the checkpoint may be left behind"
        );
        assert_eq!(read::<7>(&path).trials_used, 1500);
        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn staging_never_collides_with_the_destination_or_a_sibling() {
        let ckpt = Path::new("/runs/mpx264_s42.ckpt");
        let already_partial = Path::new("/runs/mpx264_s42.partial");
        let sibling = Path::new("/runs/mpx264_s42.backup");

        assert_ne!(staging_path(ckpt), ckpt.to_path_buf());
        assert_ne!(
            staging_path(already_partial),
            already_partial.to_path_buf(),
            "a destination already ending in .partial must not stage onto itself"
        );
        assert_ne!(
            staging_path(ckpt),
            staging_path(sibling),
            "destinations sharing a stem must not stage onto one file"
        );
        assert_eq!(staging_path(ckpt).parent(), ckpt.parent());
    }

    fn truncate_after_last_space(text: &str) -> String {
        let cut = text.trim_end().rfind(' ').unwrap();
        text[..=cut].to_string()
    }

    #[test]
    fn a_checkpoint_cut_short_is_refused_rather_than_read() {
        let mut original = state();
        original.agent.replay = Some(vec![ReplaySample {
            state: Perception::new([Symbol::Token(b'1'); 7]),
            action: 1,
            reward: 1000.0,
            next_state: Perception::new([Symbol::Token(b'0'); 7]),
            done: true,
        }]);
        let rendered = render(&original);

        // Cutting the file immediately after the space before the final `done` flag
        // still yields a field -- an empty one. Read permissively that is a terminal
        // sample silently turned into a bootstrapped one.
        assert!(std::panic::catch_unwind(|| parse::<7>(&truncate_after_last_space(&rendered))).is_err());
        assert!(std::panic::catch_unwind(|| parse::<7>(rendered.trim_end_matches("end\n"))).is_err());
        assert!(std::panic::catch_unwind(|| parse::<7>(&format!("{rendered}c extra\n"))).is_err());
        assert!(std::panic::catch_unwind(|| parse::<7>(&rendered.replace(" 1\nend", " 2\nend"))).is_err());
        assert!(parse::<7>(&rendered).agent.replay.unwrap()[0].done);
    }

    #[test]
    fn a_population_count_that_does_not_match_its_records_is_refused() {
        let rendered = render(&state());
        assert!(std::panic::catch_unwind(|| parse::<7>(&rendered.replace("population 2", "population 3"))).is_err());
        assert!(std::panic::catch_unwind(|| parse::<7>(&rendered.replace("population 2", "population 1"))).is_err());
    }
}
