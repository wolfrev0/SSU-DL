use dlrs::{
	graph::computation_graph::ComputationGraph,
	misc::util::is_equal,
	operation::{fully_connected, fully_connected_back, relu, relu_back},
};
use ndarray::Array4;

fn main() {
	let mut g = ComputationGraph::new();

	let input = g.alloc();
	let input_data =
		Array4::<f32>::from_shape_vec((1, 1, 2, 3), vec![0., 1., 2., 3., 4., 5.]).unwrap();

	let weight = g.alloc();
	let weight_data = Array4::<f32>::from_shape_vec((1, 1, 3, 1), vec![0., 1., 2.]).unwrap();

	let fc1 = g.alloc();
	g.adj[fc1].op = (fully_connected, fully_connected_back);
	g.connect(input, fc1);
	g.connect(weight, fc1);

	let (res, _) = g.run(vec![
		(input, input_data.clone()),
		(weight, weight_data.clone()),
	]);
	dbg!(input_data.clone(), weight_data.clone(), res[fc1].clone());
	assert!(is_equal(res[fc1].iter(), [5., 14.].iter()));
}
