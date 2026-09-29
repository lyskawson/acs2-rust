use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::Command;

use acs2_core::goal::Goal;
use acs2_core::rng::ChaChaRandomSource;
use acs2_envs::goal::bit_flipping::BitFlipping;
use acs2_envs::goal::hand_eye::position_goal;
use acs2_envs::goal::maze::{Coordinates, NeighbourPerception};
use acs2_envs::goal::taxi::Taxi;
use acs2_envs::maze::topology::MazeTopology;
use acs2_envs::roles::ResearchTask;
use acs2_measure::runner::{run_with_sink, AgentKind, RunMetadata, RunSettings};
use acs2_measure::task::{
    bit_goal, parse_cells, taxi_goal, BitTask, HandEyeTask, MazeTask, Task, TaxiTask,
};

mod output;

use output::output_path;

struct Options {
    task: String,
    cap: u32,
    pool: String,
    encoding: String,
    agents: Vec<AgentKind>,
    seeds: Vec<u64>,
    targets: Vec<u64>,
    out: PathBuf,
    record_starts: bool,
}

impl Options {
    fn parse() -> Self {
        let mut task = None;
        let mut cap = None;
        let mut pool = "full".to_owned();
        let mut encoding = "coordinates".to_owned();
        let mut agents = vec![AgentKind::Acs2, AgentKind::Acs2Er];
        let mut seeds = vec![42, 43, 44, 45, 46];
        let mut targets = None;
        let mut out = None;
        let mut record_starts = false;
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            if flag == "--per-start" {
                record_starts = true;
                continue;
            }
            let value = args
                .next()
                .unwrap_or_else(|| panic!("{flag} needs a value"));
            match flag.as_str() {
                "--task" => task = Some(value),
                "--cap" => cap = Some(value.parse().expect("positive cap")),
                "--pool" => pool = value,
                "--encoding" => encoding = value,
                "--agents" => {
                    agents = value
                        .split(',')
                        .map(|part| match part {
                            "acs2" => AgentKind::Acs2,
                            "acs2er" => AgentKind::Acs2Er,
                            _ => panic!("unknown agent {part}"),
                        })
                        .collect()
                }
                "--seeds" => {
                    seeds = value
                        .split(',')
                        .map(|part| part.parse().expect("seed"))
                        .collect()
                }
                "--targets" => {
                    targets = Some(
                        value
                            .split(',')
                            .map(|part| part.parse().expect("step point"))
                            .collect(),
                    )
                }
                "--out" => out = Some(PathBuf::from(value)),
                _ => panic!("unknown option {flag}"),
            }
        }
        let task = task.expect("--task is required");
        let cap = cap.expect("--cap is required");
        assert!(cap > 0);
        let targets = targets.unwrap_or_else(|| {
            if task.starts_with("handeye") {
                vec![10_000, 20_000, 50_000, 100_000]
            } else {
                vec![10_000, 20_000, 50_000, 100_000, 200_000]
            }
        });
        assert!(
            !targets.is_empty()
                && targets[0] > 0
                && targets.windows(2).all(|pair| pair[0] < pair[1])
        );
        assert!(!seeds.is_empty() && !agents.is_empty());
        Self {
            task,
            cap,
            pool,
            encoding,
            agents,
            seeds,
            targets,
            out: out.expect("--out is required"),
            record_starts,
        }
    }
}

fn execute<T, const S: usize, const G: usize, const M: usize>(
    task: &T,
    options: &Options,
    writer: &mut BufWriter<File>,
    metadata: &RunMetadata<'_>,
) where
    T: Task<S, G, M>,
{
    for &seed in &options.seeds {
        for &kind in &options.agents {
            run_with_sink(
                task,
                kind,
                seed,
                &options.targets,
                metadata,
                RunSettings {
                    evaluate_points: true,
                    capture_final_state: false,
                },
                &mut |row| {
                    writeln!(writer, "{}", row).expect("write result");
                    writer.flush().expect("flush result");
                },
            );
        }
    }
}

fn maze<const G: usize, const M: usize, E: acs2_envs::goal::maze::MazeGoalEncoding<G>>(
    geometry: &'static acs2_envs::maze::geometries::MazeGeometry,
    options: &Options,
    writer: &mut BufWriter<File>,
    metadata: &RunMetadata<'_>,
    encoding: &'static str,
) {
    let topology = MazeTopology::new(geometry).expect("valid geometry");
    let pool = if options.pool == "full" {
        topology.walkable_cells().to_vec()
    } else {
        parse_cells(&options.pool)
    };
    let task = MazeTask::<E, G>::new(options.task.clone(), geometry, pool, options.cap, encoding);
    execute::<_, 8, G, M>(&task, options, writer, metadata);
}

fn handeye<const SIDE: usize, const S: usize, const M: usize>(
    options: &Options,
    writer: &mut BufWriter<File>,
    metadata: &RunMetadata<'_>,
) {
    assert_eq!(options.encoding, "coordinates");
    let pool: Vec<Goal<2>> = if options.pool == "full" {
        (0..SIDE * SIDE)
            .map(|i| position_goal((i % SIDE, i / SIDE)))
            .collect()
    } else {
        parse_cells(&options.pool)
            .into_iter()
            .map(|(row, col)| position_goal((col, row)))
            .collect()
    };
    assert!(!pool.is_empty());
    let task = HandEyeTask::<SIDE, S>::new(options.task.clone(), options.cap, pool);
    execute::<_, S, 2, M>(&task, options, writer, metadata);
}

fn bit<const N: usize, const M: usize>(
    options: &Options,
    writer: &mut BufWriter<File>,
    metadata: &RunMetadata<'_>,
) {
    assert!(options.encoding == "coordinates" || options.encoding == "bits");
    let template =
        BitFlipping::<N>::with_step_cap(options.cap, Box::new(ChaChaRandomSource::from_seed(0)));
    let pool = if options.pool == "full" {
        template.goal_pool().collect()
    } else {
        options.pool.split(',').map(bit_goal::<N>).collect()
    };
    let task = BitTask::<N>::new(options.cap, pool);
    execute::<_, N, N, M>(&task, options, writer, metadata);
}

fn command_text(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

fn hardware() -> (String, String) {
    let host = command_text("hostname", &[]).unwrap_or_else(|| "unknown".to_owned());
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|value| value.split_once(':'))
                    .map(|(_, name)| name.trim().to_owned())
            })
        })
        .or_else(|| command_text("sysctl", &["-n", "machdep.cpu.brand_string"]))
        .unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    (host, cpu)
}

fn main() {
    let options = Options::parse();
    let path = output_path(&options.out);
    let file = File::create(&path).expect("create results");
    let mut writer = BufWriter::new(file);
    let (host, cpu_model) = hardware();
    let metadata = RunMetadata {
        commit: env!("ACS2_BUILD_COMMIT"),
        source_state: env!("ACS2_BUILD_SOURCE_STATE"),
        host: &host,
        cpu_model: &cpu_model,
        record_starts: options.record_starts,
    };
    match ResearchTask::named(&options.task).expect("registered task") {
        ResearchTask::GoalMaze(geometry) => match options.encoding.as_str() {
            "coordinates" => maze::<2, 10, Coordinates>(
                geometry,
                &options,
                &mut writer,
                &metadata,
                "coordinates",
            ),
            "perception" => maze::<8, 16, NeighbourPerception>(
                geometry,
                &options,
                &mut writer,
                &metadata,
                "perception",
            ),
            _ => panic!("unknown maze encoding"),
        },
        ResearchTask::HandEye(3) => handeye::<3, 10, 12>(&options, &mut writer, &metadata),
        ResearchTask::HandEye(4) => handeye::<4, 17, 19>(&options, &mut writer, &metadata),
        ResearchTask::HandEye(5) => handeye::<5, 26, 28>(&options, &mut writer, &metadata),
        ResearchTask::HandEye(_) => unreachable!(),
        ResearchTask::Taxi => {
            assert!(options.encoding == "coordinates" || options.encoding == "stand");
            let template = Taxi::new(options.cap, Box::new(ChaChaRandomSource::from_seed(0)));
            let pool = if options.pool == "full" {
                template.goal_pool().to_vec()
            } else {
                options
                    .pool
                    .split(',')
                    .map(|part| taxi_goal(part.parse().expect("stand index")))
                    .collect()
            };
            let task = TaxiTask::new(options.cap, pool);
            execute::<_, 3, 1, 4>(&task, &options, &mut writer, &metadata);
        }
        ResearchTask::BitFlipping(n) => match n {
            1 => bit::<1, 2>(&options, &mut writer, &metadata),
            2 => bit::<2, 4>(&options, &mut writer, &metadata),
            3 => bit::<3, 6>(&options, &mut writer, &metadata),
            4 => bit::<4, 8>(&options, &mut writer, &metadata),
            5 => bit::<5, 10>(&options, &mut writer, &metadata),
            6 => bit::<6, 12>(&options, &mut writer, &metadata),
            7 => bit::<7, 14>(&options, &mut writer, &metadata),
            8 => bit::<8, 16>(&options, &mut writer, &metadata),
            9 => bit::<9, 18>(&options, &mut writer, &metadata),
            10 => bit::<10, 20>(&options, &mut writer, &metadata),
            11 => bit::<11, 22>(&options, &mut writer, &metadata),
            12 => bit::<12, 24>(&options, &mut writer, &metadata),
            13 => bit::<13, 26>(&options, &mut writer, &metadata),
            14 => bit::<14, 28>(&options, &mut writer, &metadata),
            15 => bit::<15, 30>(&options, &mut writer, &metadata),
            16 => bit::<16, 32>(&options, &mut writer, &metadata),
            _ => unreachable!(),
        },
    }
}
