use std::collections::VecDeque;

use ndarray::Array4;

use crate::operation::{identity, identity_back};

pub struct Node {
	pub id: usize,
	pub succ: Vec<(usize, usize)>, //(successor x, adj[x]에서 id의 index)
	pub pred: Vec<(usize, usize)>, //(successor x, adj[x]에서 id의 index)
	pub op: (
		fn(&Vec<Array4<f32>>) -> Array4<f32>,      //forward prop
		fn(&Vec<Array4<f32>>) -> Vec<Array4<f32>>, //backward prop
	),
}
impl Node {
	pub fn is_input(&self) -> bool {
		self.id == 0
	}
	pub fn is_terminal(&self) -> bool {
		self.pred.is_empty()
	}
}

pub struct ComputationGraph {
	pub adj: Vec<Node>,
}

//source node: 0
//pred==empty but not source node: data node(constant)
impl ComputationGraph {
	pub fn new() -> ComputationGraph {
		Self { adj: Vec::new() }
	}
	pub fn alloc(&mut self) -> usize {
		let id = self.adj.len();
		let x = Node {
			id,
			succ: Vec::new(),
			pred: Vec::new(),
			op: (identity, identity_back),
		};
		self.adj.push(x);
		id
	}

	pub fn connect(&mut self, x: usize, y: usize) {
		let xi = self.adj[y].pred.len();
		let yi = self.adj[x].pred.len();
		self.adj[x].succ.push((y, xi));
		self.adj[y].pred.push((x, yi));
	}

	fn topological_order() -> Vec<usize> {
		todo!();
	}

	//return (outputs, gradients) of current graph
	pub fn run(
		&self,
		terminal_init: Vec<(usize, Array4<f32>)>,
	) -> (Vec<Array4<f32>>, Vec<Array4<f32>>) {
		let mut snk = usize::MAX;
		let mut dp_out = vec![Option::None; self.adj.len()];
		for (i, x) in terminal_init {
			assert!(
				self.adj[i].is_terminal(),
				"Only terminal nodes can have initial value"
			);
			assert!(
				self.adj[i].op == (identity, identity_back),
				"Operation of terminal node should identity"
			);
			dp_out[i] = Some(self.adj[i].op.0(&vec![x; 1]).clone())
		}
		for i in 0..self.adj.len() {
			if self.adj[i].succ.len() == 0 {
				assert!(snk == usize::MAX, "snk should be unique");
				snk = i;
			}
			if self.adj[i].is_terminal() {
				assert!(
					dp_out[i].is_some(),
					"Terminal nodes should have initial value"
				);
			}
		}
		assert!(snk != usize::MAX, "snk should exist");
		self.get_outputs_dfs(&mut dp_out, snk);

		let mut dp_grad = vec![Option::None; self.adj.len()];

		let output_snk_shape = dp_out[snk].as_ref().unwrap().shape();
		let output_shape = (
			output_snk_shape[0],
			output_snk_shape[1],
			output_snk_shape[2],
			output_snk_shape[3],
		);
		dp_grad[snk] = dp_out[snk].clone(); //Some(Array4::ones(output_shape));

		//a.k.a backward propagation
		//TODO: BFS won't be fit when graph is not tree. (ex: residual block)
		//use topological_order()
		let mut q = VecDeque::new();
		q.push_back(snk);
		while let Some(x) = q.pop_front() {
			if self.adj[x].is_terminal() {
				continue;
			}
			let mut input = Vec::new();
			for (y, _) in self.adj[x].pred.iter() {
				input.push(self.get_outputs_dfs(&mut dp_out, *y));
			}
			input.push(dp_grad[x].to_owned().unwrap());
			let input_grads = self.adj[x].op.1(&input);
			for i in 0..self.adj[x].pred.len() {
				let y = self.adj[x].pred[i].0;
				dp_grad[y] = match &dp_grad[y] {
					None => Some(input_grads[i].clone()),
					Some(val) => Some(val + &input_grads[i]),
				};
				q.push_back(y);
			}
		}

		let outputs = Vec::from_iter(dp_out.into_iter().filter_map(|x| x));
		let gradients = Vec::from_iter(dp_grad.into_iter().filter_map(|x| x));
		(outputs, gradients)
	}
	//a.k.a forward propagation
	fn get_outputs_dfs(&self, dp_out: &mut Vec<Option<Array4<f32>>>, id: usize) -> Array4<f32> {
		match &dp_out[id] {
			Some(ret) => ret.clone(),
			None => {
				let mut input = Vec::new();
				for (x, _) in self.adj[id].pred.iter() {
					input.push(self.get_outputs_dfs(dp_out, *x));
				}
				let output = self.adj[id].op.0(&input);
				dp_out[id] = Some(output.clone());
				dp_out[id].to_owned().unwrap()
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use ndarray::Array4;

	use crate::{
		graph::computation_graph::ComputationGraph,
		misc::util::is_equal,
		operation::{
			fully_connected, fully_connected_back, relu, relu_back, softmax_cross_entropy,
			softmax_cross_entropy_back,
		},
	};

	#[test]
	fn test_relu() {
		let mut g = ComputationGraph::new();

		let input = g.alloc();
		let input_data =
			Array4::<f32>::from_shape_vec((1, 1, 2, 3), vec![0., 1., -2., 3., -4., 5.]).unwrap();

		let hidden = g.alloc();
		g.adj[hidden].op = (relu, relu_back);
		g.connect(input, hidden);

		let (out, grad) = g.run(vec![(input, input_data.clone())]);
		assert!(is_equal(
			out[hidden].iter(),
			[0., 1., 0., 3., 0., 5.].iter()
		));
		assert!(is_equal(
			grad[input].iter(),
			[0., 1., 0., 1., 0., 1.].iter()
		));
	}

	#[test]
	fn test_matmul() {
		/* REFERENCE CODE
		import torch
		A = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
		B = torch.arange(3, dtype=torch.float32).reshape(3, 1).requires_grad_()
		C = torch.mm(A, B)
		loss = C.sum()
		loss.backward()
		grad_A = A.grad
		grad_B = B.grad
		print("Matrix A:")
		print(A)
		print("\nMatrix B:")
		print(B)
		print("\nMatrix C (result of A*B):")
		print(C)
		print("\nGradient of A:")
		print(grad_A)
		print("\nGradient of B:")
		print(grad_B)*/
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
		assert!(is_equal(res[fc2].iter(), [19.].iter()));
		assert!(is_equal(grad[fc2].iter(), [1.].iter()));

		assert!(is_equal(res[fc1].iter(), [5., 14.].iter()));
		assert!(is_equal(grad[fc1].iter(), [1., 1.].iter()));

		assert!(is_equal(
			grad[input].iter(),
			[0., 1., 2., 0., 1., 2.].iter()
		));
		assert!(is_equal(grad[weight1].iter(), [3., 5., 7.].iter()));
		assert!(is_equal(grad[weight2].iter(), [5., 14.].iter()));
	}

	#[test]
	fn fc_relu_fc_relu() {
		/*REFERENCE CODE
		import torch
		import torch.nn as nn
		torch.manual_seed(11111)
		class SimpleNet(nn.Module):
			def __init__(self, input_size, hidden_size, output_size):
				super(SimpleNet, self).__init__()
				self.fc1 = nn.Linear(input_size, hidden_size, False)
				self.relu1 = nn.ReLU()
				self.fc2 = nn.Linear(hidden_size, output_size, False)
				self.relu2 = nn.ReLU()

			def forward(self, x):
				x = self.relu1(self.fc1(x))
				x = self.relu2(self.fc2(x))
				return x

		input_size = 8
		hidden_size = 6
		output_size = 4

		model = SimpleNet(input_size, hidden_size, output_size)
		example_input = torch.rand((1, input_size))
		output = model(example_input)

		print(model)
		print("\nInput:")
		print(example_input)
		print("\nWeights of Linear Layer 1:")
		print(model.fc1.weight)
		print("\nWeights of Linear Layer 2:")
		print(model.fc2.weight)
		print("\nOutput:")
		print(output)

		output.backward(gradient=torch.tensor([[1.,1.,1.,1.]]))
		print("\nGradients of Linear Layer 1 weights:")
		print(model.fc1.weight.grad)
		print("\nGradients of Linear Layer 2 weights:")
		print(model.fc2.weight.grad)*/
		let mut g = ComputationGraph::new();

		let input = g.alloc();
		let input_data = Array4::<f32>::from_shape_vec(
			(1, 1, 8, 1),
			vec![
				0.7608, 0.9041, 0.9830, 0.3026, 0.3139, 0.1113, 0.2464, 0.9967,
			],
		)
		.unwrap();

		let weight1 = g.alloc();
		let weight1_data = Array4::<f32>::from_shape_vec(
			(1, 1, 6, 8),
			vec![
				0.1806, 0.0927, 0.0754, -0.1094, 0.0388, -0.2083, -0.3351, -0.2591, 0.2947,
				-0.2373, 0.2207, -0.2163, 0.0163, 0.1468, -0.3526, 0.3319, -0.2321, 0.1037,
				-0.2045, -0.1501, 0.2676, 0.3087, 0.3164, -0.0726, -0.1910, 0.2309, -0.2953,
				-0.0291, -0.1337, 0.2855, -0.0138, 0.0819, 0.0396, 0.1394, 0.2698, -0.1355,
				-0.1322, -0.2800, -0.3354, 0.1577, 0.3326, -0.1080, 0.0047, -0.1543, 0.0193,
				-0.0340, -0.3209, 0.0165,
			],
		)
		.unwrap();

		let weight2 = g.alloc();
		let weight2_data = Array4::<f32>::from_shape_vec(
			(1, 1, 4, 6),
			vec![
				0.1992, 0.1353, -0.3798, -0.1454, 0.3411, 0.0315, -0.3327, -0.1407, -0.1425,
				0.1139, -0.3173, 0.1386, 0.2275, -0.1266, -0.0111, 0.1349, 0.2610, -0.1336,
				-0.2452, 0.0849, 0.2613, 0.2177, -0.0315, -0.0740,
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
		for i in grad.iter() {
			println!("{}", i);
		}
		assert!(is_equal(
			res[relu2].iter(),
			[0.1898, 0.0000, 0.0387, 0.0203].iter()
		));
		assert!(is_equal(
			grad[weight1].iter(),
			[
				0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0712, 0.0846,
				0.0920, 0.0283, 0.0294, 0.0104, 0.0231, 0.0933, 0.0000, 0.0000, 0.0000, 0.0000,
				0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
				0.0000, 0.0000, 0.4341, 0.5159, 0.5609, 0.1727, 0.1791, 0.0635, 0.1406, 0.5687,
				-0.1339, -0.1592, -0.1730, -0.0533, -0.0553, -0.0196, -0.0434, -0.1755
			]
			.iter()
		));
		assert!(is_equal(
			grad[weight2].iter(),
			[
				0.0000, 0.4265, 0.0000, 0.0000, 0.3823, 0.0529, 0.0000, 0.0000, 0.0000, 0.0000,
				0.0000, 0.0000, 0.0000, 0.4265, 0.0000, 0.0000, 0.3823, 0.0529, 0.0000, 0.4265,
				0.0000, 0.0000, 0.3823, 0.0529
			]
			.iter()
		));
	}

	#[test]
	fn softmax() {}
	#[test]
	fn cross_entropy() {}
	#[test]
	fn softmax_cross_entropy_test() {
		/*REFERENCE CODE
		import torch
		import torch.nn.functional as F
		torch.manual_seed(123)
		num_classes = 3
		logits = torch.randn(1, num_classes, requires_grad=True)
		target = torch.randint(0, num_classes, (1,))
		softmax_output = F.softmax(logits, dim=1)
		cross_entropy_loss = F.cross_entropy(logits, target)
		cross_entropy_loss.backward()
		gradients = logits.grad
		print("Logits:", logits)
		print("Softmax output:", softmax_output)
		print("Target:", target)
		print("Cross Entropy Loss:", cross_entropy_loss.item())
		print("Gradients:", gradients)*/
		let mut g = ComputationGraph::new();

		let input = g.alloc();
		let input_data =
			Array4::<f32>::from_shape_vec((1, 1, 3, 1), vec![-0.1115, 0.1204, -0.3696]).unwrap();

		let truth = g.alloc();
		let truth_data = Array4::<f32>::from_shape_vec((1, 1, 3, 1), vec![0., 0., 1.]).unwrap();

		let sc = g.alloc();
		g.adj[sc].op = (softmax_cross_entropy, softmax_cross_entropy_back);
		g.connect(input, sc);
		g.connect(truth, sc);

		let (res, grad) = g.run(vec![
			(input, input_data.clone()),
			(truth, truth_data.clone()),
		]);
		for i in res.iter() {
			println!("{}", i);
		}
		for i in grad.iter() {
			println!("{}", i);
		}
		assert!(is_equal(res[sc].iter(), [1.3678419589996338].iter()));
		assert!(is_equal(
			grad[input].iter(),
			[0.3297, 0.4157, -0.7453].iter()
		));
	}
}
