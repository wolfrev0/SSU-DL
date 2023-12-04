use crate::{
	computation_graph::ComputationGraph,
	operation::{
		eltw_mult_bwd, eltw_mult_fwd, matmul_bwd, matmul_fwd, softmax_y_bwd, softmax_y_fwd,
		transpose_bwd, transpose_fwd,
	},
};

pub fn build_attention(
	g: &mut ComputationGraph,
	input: usize,
	wq: usize,
	wk: usize,
	wv: usize,
	rsqrt: usize,
) -> usize {
	let q = g.alloc();
	g.adj[q].op = (matmul_fwd, matmul_bwd);
	g.connect(wq, q);
	g.connect(input, q);

	let k = g.alloc();
	g.adj[k].op = (matmul_fwd, matmul_bwd);
	g.connect(wk, k);
	g.connect(input, k);

	let v = g.alloc();
	g.adj[v].op = (matmul_fwd, matmul_bwd);
	g.connect(wv, v);
	g.connect(input, v);

	let tp = g.alloc();
	g.adj[tp].op = (transpose_fwd, transpose_bwd);
	g.connect(k, tp);

	let kq = g.alloc();
	g.adj[kq].op = (matmul_fwd, matmul_bwd);
	g.connect(tp, kq);
	g.connect(q, kq);

	let mul_rsqrt = g.alloc();
	g.adj[mul_rsqrt].op = (eltw_mult_fwd, eltw_mult_bwd);
	g.connect(kq, mul_rsqrt);
	g.connect(rsqrt, mul_rsqrt);

	let attw = g.alloc();
	g.adj[attw].op = (softmax_y_fwd, softmax_y_bwd);
	g.connect(mul_rsqrt, attw);

	let atts = g.alloc();
	g.adj[atts].op = (matmul_fwd, matmul_bwd);
	g.connect(v, atts);
	g.connect(attw, atts);

	atts
}
