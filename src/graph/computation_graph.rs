use ndarray::Array4;

use crate::operation::identity;

pub struct Node {
	pub id: usize,
	pub succ: Vec<usize>,
	pub pred: Vec<usize>,
	pub op: (
		fn(&Vec<Array4<f32>>) -> Vec<Array4<f32>>, //forward prop
		fn(&Vec<Array4<f32>>) -> Vec<Array4<f32>>, //backward prop
	),
}
impl Node {
	pub fn is_input(&self) -> bool {
		self.id == 0
	}
	//NOTE: ideally, if an operation all input is constant is also constant,
	//but we can precalculate such case into one node.
	//So checking pred.empty() is enough as of now.
	pub fn is_constant(&self) -> bool {
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
			op: (identity, identity),
		};
		self.adj.push(x);
		id
	}
	pub fn connect(&mut self, x: usize, y: usize) {
		self.adj[x].succ.push(y);
		self.adj[y].pred.push(x);
	}
	pub fn run(&self, init: Vec<(usize, Array4<f32>)>) -> Vec<Array4<f32>> {
		let mut snk = usize::MAX;
		let mut dp = vec![Option::None; self.adj.len()];
		for (i, x) in init {
			assert!(
				self.adj[i].is_constant(),
				"Only constant nodes can have initial value"
			);
			dp[i] = Some(self.adj[i].op.0(&vec![x; 1])[0].clone())
		}
		for i in 0..self.adj.len() {
			if self.adj[i].succ.len() == 0 {
				assert!(snk == usize::MAX, "snk should be unique");
				snk = i;
			}
			if self.adj[i].is_constant() {
				assert!(dp[i].is_some(), "Constant nodes should have initial value");
			}
		}
		assert!(snk != usize::MAX, "snk should exist");
		self.run_dfs(&mut dp, snk);
		Vec::from_iter(dp.into_iter().filter_map(|x| x))
	}
	fn run_dfs(&self, dp: &mut Vec<Option<Array4<f32>>>, id: usize) -> Array4<f32> {
		match &dp[id] {
			Some(x) => x.clone(),
			None => {
				let mut input = Vec::new();
				for y in self.adj[id].pred.iter() {
					input.push(self.run_dfs(dp, *y));
				}
				let output = self.adj[id].op.0(&input);
				dp[id] = Some(output.first().unwrap().clone());
				dp[id].to_owned().unwrap()
			}
		}
	}
}
