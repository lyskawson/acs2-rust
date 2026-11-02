use acs2_envs::maze::topology::MazeTopology;
use acs2_envs::roles::ResearchTask;

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(args.len(), 2, "expected maze name and evaluation cap");
    let ResearchTask::GoalMaze(geometry) = ResearchTask::named(&args[0]).expect("known maze")
    else {
        panic!("ground truth requires a maze")
    };
    let topology = MazeTopology::new(geometry).expect("valid environment topology");
    println!(
        "{}",
        acs2_measure::maze_truth::model(&topology, &args[0], args[1].parse().expect("cap"))
    );
}
