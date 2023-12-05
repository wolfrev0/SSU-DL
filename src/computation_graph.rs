use std::collections::VecDeque;

use ndarray::Array4;

use crate::operation::{identity_bwd, identity_fwd};

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
			op: (identity_fwd, identity_bwd),
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

	fn topological_order(&self) -> Vec<usize> {
		let n = self.adj.len();
		let mut ret = Vec::new();
		let mut deg = (0..n).map(|x| self.adj[x].pred.len()).collect::<Vec<_>>();
		let mut q = VecDeque::new();
		for i in 0..n {
			if deg[i] == 0 {
				q.push_back(i);
			}
		}
		while let Some(x) = q.pop_front() {
			ret.push(x);
			for (y, _) in self.adj[x].succ.iter() {
				deg[*y] -= 1;
				if deg[*y] == 0 {
					q.push_back(*y);
				}
			}
		}
		ret
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
				self.adj[i].op == (identity_fwd, identity_bwd),
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
		dp_grad[snk] = Some(Array4::ones(output_shape));

		//a.k.a backward propagation
		//TODO: BFS won't be fit when graph is not tree. (ex: residual block)
		//use topological_order()
		let ord = {
			let mut tmp = self.topological_order();
			tmp.reverse();
			tmp
		};
		for x in ord {
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
