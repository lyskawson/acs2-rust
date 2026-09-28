use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use acs2_core::goal::Goal;
use acs2_core::rng::ChaChaRandomSource;
use acs2_envs::goal::bit_flipping::BitFlipping;
use acs2_envs::goal::hand_eye::position_goal;
use acs2_envs::goal::maze::{Coordinates, NeighbourPerception};
use acs2_envs::goal::taxi::Taxi;
use acs2_envs::maze::topology::MazeTopology;
use acs2_envs::roles::ResearchTask;
use acs2_measure::runner::{run, AgentKind};
use acs2_measure::task::{
    bit_goal, parse_cells, taxi_goal, BitTask, HandEyeTask, MazeTask, Task, TaxiTask,
};

struct Options {
    task: String,
    cap: u32,
    pool: String,
    encoding: String,
    agents: Vec<AgentKind>,
    seeds: Vec<u64>,
    targets: Vec<u64>,
    out: PathBuf,
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
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
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
        }
    }
}

fn output_path(out: &Path) -> PathBuf {
    let absolute = if out.is_absolute() {
        out.to_path_buf()
    } else {
        std::env::current_dir().unwrap().join(out)
    };
    let parent = absolute.parent().expect("output parent");
    std::fs::create_dir_all(parent).expect("create output directory");
    let parent = parent.canonicalize().expect("output parent exists");
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .canonicalize()
        .unwrap();
    assert!(
        !parent.starts_with(repo),
        "results belong outside the checkout"
    );
    parent.join(absolute.file_name().expect("output file name"))
}

fn execute<T, const S: usize, const G: usize, const M: usize>(
    task: &T,
    options: &Options,
    writer: &mut BufWriter<File>,
    commit: &str,
) where
    T: Task<S, G, M>,
{
    for &seed in &options.seeds {
        for &kind in &options.agents {
            let output = run(task, kind, seed, &options.targets, commit, true);
            for row in output.rows {
                writeln!(writer, "{}", row).expect("write result");
            }
            writer.flush().expect("flush result");
        }
    }
}

fn maze<const G: usize, const M: usize, E: acs2_envs::goal::maze::MazeGoalEncoding<G>>(
    geometry: &'static acs2_envs::maze::geometries::MazeGeometry,
    options: &Options,
    writer: &mut BufWriter<File>,
    commit: &str,
    encoding: &'static str,
) {
    let topology = MazeTopology::new(geometry).expect("valid geometry");
    let pool = if options.pool == "full" {
        topology.walkable_cells().to_vec()
    } else {
        parse_cells(&options.pool)
    };
    let task = MazeTask::<E, G>::new(options.task.clone(), geometry, pool, options.cap, encoding);
    execute::<_, 8, G, M>(&task, options, writer, commit);
}

fn handeye<const SIDE: usize, const S: usize, const M: usize>(
    options: &Options,
    writer: &mut BufWriter<File>,
    commit: &str,
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
    execute::<_, S, 2, M>(&task, options, writer, commit);
}

fn bit<const N: usize, const M: usize>(
    options: &Options,
    writer: &mut BufWriter<File>,
    commit: &str,
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
    execute::<_, N, N, M>(&task, options, writer, commit);
}

fn main() {
    let options = Options::parse();
    let path = output_path(&options.out);
    let file = File::create(&path).expect("create results");
    let mut writer = BufWriter::new(file);
    let commit = String::from_utf8(
        Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("git commit")
            .stdout,
    )
    .unwrap();
    let commit = commit.trim();
    match ResearchTask::named(&options.task).expect("registered task") {
        ResearchTask::GoalMaze(geometry) => match options.encoding.as_str() {
            "coordinates" => {
                maze::<2, 10, Coordinates>(geometry, &options, &mut writer, commit, "coordinates")
            }
            "perception" => maze::<8, 16, NeighbourPerception>(
                geometry,
                &options,
                &mut writer,
                commit,
                "perception",
            ),
            _ => panic!("unknown maze encoding"),
        },
        ResearchTask::HandEye(3) => handeye::<3, 10, 12>(&options, &mut writer, commit),
        ResearchTask::HandEye(4) => handeye::<4, 17, 19>(&options, &mut writer, commit),
        ResearchTask::HandEye(5) => handeye::<5, 26, 28>(&options, &mut writer, commit),
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
            execute::<_, 3, 1, 4>(&task, &options, &mut writer, commit);
        }
        ResearchTask::BitFlipping(n) => match n {
            1 => bit::<1, 2>(&options, &mut writer, commit),
            2 => bit::<2, 4>(&options, &mut writer, commit),
            3 => bit::<3, 6>(&options, &mut writer, commit),
            4 => bit::<4, 8>(&options, &mut writer, commit),
            5 => bit::<5, 10>(&options, &mut writer, commit),
            6 => bit::<6, 12>(&options, &mut writer, commit),
            7 => bit::<7, 14>(&options, &mut writer, commit),
            8 => bit::<8, 16>(&options, &mut writer, commit),
            9 => bit::<9, 18>(&options, &mut writer, commit),
            10 => bit::<10, 20>(&options, &mut writer, commit),
            11 => bit::<11, 22>(&options, &mut writer, commit),
            12 => bit::<12, 24>(&options, &mut writer, commit),
            13 => bit::<13, 26>(&options, &mut writer, commit),
            14 => bit::<14, 28>(&options, &mut writer, commit),
            15 => bit::<15, 30>(&options, &mut writer, commit),
            16 => bit::<16, 32>(&options, &mut writer, commit),
            _ => unreachable!(),
        },
    }
}
