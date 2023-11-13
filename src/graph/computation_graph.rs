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

	//return (outputs, gradients) of current graph
	pub fn run(&self, init: Vec<(usize, Array4<f32>)>) -> (Vec<Array4<f32>>, Vec<Array4<f32>>) {
		let mut snk = usize::MAX;
		let mut dp_out = vec![Option::None; self.adj.len()];
		for (i, x) in init {
			assert!(
				self.adj[i].is_terminal(),
				"Only terminal nodes can have initial value"
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
		dp_grad[snk] = Some(Array4::ones(output_shape));

		//a.k.a backward propagation
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
		operation::{fully_connected, fully_connected_back, relu, relu_back},
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
		torch.manual_seed(0)

		# Create two matrices with sequential values
		A = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
		B = torch.arange(3, dtype=torch.float32).reshape(3, 1).requires_grad_()

		# Matrix multiplication
		C = torch.mm(A, B)

		# Define a scalar loss function (for example, the sum of all elements in C)
		loss = C.sum()

		# Compute gradients
		loss.backward()

		# Access the gradients
		grad_A = A.grad
		grad_B = B.grad

		# Print the results
		print("Matrix A:")
		print(A)
		print("\nMatrix B:")
		print(B)
		print("\nMatrix C (result of A*B):")
		print(C)
		print("\nGradient of A:")
		print(grad_A)
		print("\nGradient of B:")
		print(grad_B) */
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
		torch.manual_seed(0)

		# Define the neural network class
		class SimpleNet(nn.Module):
			def __init__(self, input_size, hidden_size, output_size):
				super(SimpleNet, self).__init__()

				# Linear layer 1
				self.fc1 = nn.Linear(input_size, hidden_size, False)
				# ReLU activation 1
				self.relu1 = nn.ReLU()

				# Linear layer 2
				self.fc2 = nn.Linear(hidden_size, output_size, False)
				# ReLU activation 2
				self.relu2 = nn.ReLU()

			def forward(self, x):
				# Forward pass
				x = self.relu1(self.fc1(x))
				x = self.relu2(self.fc2(x))
				return x

		# Create an instance of the SimpleNet
		input_size = 10
		hidden_size = 5
		output_size = 1

		model = SimpleNet(input_size, hidden_size, output_size)

		# Example input
		example_input = torch.rand((1, input_size))-0.2

		# Forward pass through the network
		output = model(example_input)

		# Backward pass to calculate gradients
		output.backward()

		# Print the model architecture and values
		print(model)
		print("\nInput:")
		print(example_input)
		print("\nWeights of Linear Layer 1:")
		print(model.fc1.weight)
		print("\nWeights of Linear Layer 2:")
		print(model.fc2.weight)
		print("\nOutput:")
		print(output)

		# Print gradients
		print("\nGradients of Linear Layer 1 weights:")
		print(model.fc1.weight.grad)

		print("\nGradients of Linear Layer 2 weights:")
		print(model.fc2.weight.grad)
				 */
		let mut g = ComputationGraph::new();

		let input = g.alloc();
		let input_data = Array4::<f32>::from_shape_vec(
			(1, 1, 10, 1),
			vec![
				0.0056, 0.3932, -0.0877, -0.0465, 0.0417, 0.5262, 0.5011, 0.0038, 0.4511, 0.5745,
			],
		)
		.unwrap();

		let weight1 = g.alloc();
		let weight1_data = Array4::<f32>::from_shape_vec(
			(1, 1, 5, 10),
			vec![
				-0.0024, 0.1696, -0.2603, -0.2327, -0.1218, 0.0848, -0.0063, 0.2507, -0.0281,
				0.0837, -0.0956, -0.0622, -0.3021, -0.2094, -0.1304, 0.0117, 0.1250, 0.1897,
				-0.2144, -0.1377, 0.1149, 0.2626, -0.0651, 0.2366, -0.0510, 0.0335, 0.2863,
				-0.2934, -0.1991, -0.0801, -0.1233, 0.2732, -0.2050, -0.1456, -0.2209, -0.2962,
				-0.1846, 0.2718, 0.1411, 0.1533, 0.0166, -0.1621, 0.0535, -0.2953, -0.2285,
				-0.1630, 0.1995, 0.1854, -0.1402, -0.0114,
			],
		)
		.unwrap();

		let weight2 = g.alloc();
		let weight2_data = Array4::<f32>::from_shape_vec(
			(1, 1, 1, 5),
			vec![0.2860, 0.4446, 0.1775, 0.0604, 0.2999],
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
		assert!(is_equal(res[relu2].iter(), [0.0725].iter()));
		assert!(is_equal(
			grad[weight2].iter(),
			[0.1731, 0.0000, 0.1206, 0.0266, 0.0000].iter()
		));
		assert!(is_equal(
			grad[weight1].iter(),
			[
				0.0016, 0.1125, -0.0251, -0.0133, 0.0119, 0.1505, 0.1433, 0.0011, 0.1290, 0.1643,
				0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
				0.0010, 0.0698, -0.0156, -0.0083, 0.0074, 0.0934, 0.0889, 0.0007, 0.0801, 0.1020,
				0.0003, 0.0238, -0.0053, -0.0028, 0.0025, 0.0318, 0.0303, 0.0002, 0.0273, 0.0347,
				0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
			]
			.iter()
		));
	}
}
