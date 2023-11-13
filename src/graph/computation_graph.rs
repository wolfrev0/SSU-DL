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
			let grads_local = self.adj[x].op.1(&input);
			for i in 0..self.adj[x].pred.len() {
				dp_grad[self.adj[x].pred[i].0] = match &dp_grad[self.adj[x].pred[i].0] {
					None => Some(grads_local[i].clone()),
					Some(val) => Some(val + dp_grad[x].as_ref().unwrap() * &grads_local[i]),
				};
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
	#[ignore] //fully_connected_back() not implemented
	fn test_matmul() {
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
		dbg!(input_data.clone());
		assert!(is_equal(
			out[hidden].iter(),
			[0., 1., 0., 3., 0., 5.].iter()
		));
		assert!(is_equal(
			grad[input].iter(),
			[0., 1., 0., 1., 0., 1.].iter()
		));
	}
}
