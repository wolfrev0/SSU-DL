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

	let weight1 = g.alloc();
	let weight1_data = Array4::<f32>::from_shape_vec((1, 1, 3, 1), vec![0., 1., 2.]).unwrap();

	let weight2 = g.alloc();
	let weight2_data = Array4::<f32>::from_shape_vec((1, 1, 1, 2), vec![1., 1.]).unwrap();

	let fc1 = g.alloc();
	g.adj[fc1].op = (fully_connected, fully_connected_back);
	g.connect(input, fc1);
	g.connect(weight1, fc1);

	let fc2 = g.alloc();
	g.adj[fc2].op = (fully_connected, fully_connected_back);
	g.connect(weight2, fc2);
	g.connect(fc1, fc2);

	let (res, grad) = g.run(vec![
		(input, input_data.clone()),
		(weight1, weight1_data.clone()),
		(weight2, weight2_data.clone()),
	]);
	println!("{}", res[fc2]);
	println!("{}", grad[fc2]);
	for i in grad {
		println!("grad {}", i);
	}
}
