use dlrs::{
	graph::computation_graph::ComputationGraph,
	misc::util::is_equal,
	operation::{fully_connected, fully_connected_back, relu, relu_back},
};
use ndarray::Array4;

fn main() {
	let mut g = ComputationGraph::new();

	let input = g.alloc();
	let input_data = Array4::<f32>::from_shape_vec(
		(1, 1, 10, 1),
		vec![
			-0.1871, 0.7550, 0.4201, 0.0270, 0.6867, 0.3722, 0.4565, -0.0877, -0.1779, 0.2436,
		],
	)
	.unwrap();

	let weight1 = g.alloc();
	let weight1_data = Array4::<f32>::from_shape_vec(
		(1, 1, 5, 10),
		vec![
			0.0864, 0.2842, -0.1581, -0.0496, 0.1931, -0.1462, -0.1570, 0.0602, 0.1350, -0.2531,
			-0.2693, 0.1322, 0.1294, 0.1118, 0.2032, -0.2386, -0.1488, -0.1741, 0.0037, 0.0996,
			0.2535, -0.1904, 0.2028, -0.0188, 0.2438, 0.0412, -0.0033, 0.0101, -0.2254, -0.1970,
			0.1999, 0.0695, -0.1492, 0.1013, -0.2092, 0.0947, 0.0459, 0.3104, -0.1615, -0.2611,
			0.0091, -0.0975, -0.1742, 0.0012, 0.0932, 0.0878, 0.0461, 0.1912, -0.1670, 0.0668,
		],
	)
	.unwrap();

	let weight2 = g.alloc();
	let weight2_data = Array4::<f32>::from_shape_vec(
		(1, 1, 3, 5),
		vec![
			0.1072, -0.2878, -0.4372, -0.1042, 0.2964, -0.1477, 0.2980, 0.3052, -0.3202, -0.0285,
			-0.3447, 0.1010, 0.3351, 0.3897, 0.1993,
		],
	)
	.unwrap();

	let fc1 = g.alloc();
	g.adj[fc1].op = (fully_connected, fully_connected_back);
	g.connect(weight1, fc1);
	g.connect(input, fc1);

	let relu1 = g.alloc();
	g.adj[relu1].op = (relu, relu_back);
	g.connect(fc1, relu1);

	let fc2 = g.alloc();
	g.adj[fc2].op = (fully_connected, fully_connected_back);
	g.connect(weight2, fc2);
	g.connect(relu1, fc2);

	let relu2 = g.alloc();
	g.adj[relu2].op = (relu, relu_back);
	g.connect(fc2, relu2);

	let (res, grad) = g.run(vec![
		(input, input_data.clone()),
		(weight1, weight1_data.clone()),
		(weight2, weight2_data.clone()),
	]);

	assert!(is_equal(
		res[relu2].iter(),
		[0., 0.0816276, 0.029333532].iter()
	));
}
