use crate::{
	computation_graph::ComputationGraph,
	operation::{
		concat4x_bwd, concat4x_fwd, concat4y_bwd, concat4y_fwd, div_sqrt2_bwd, div_sqrt2_fwd,
		div_sqrt50_bwd, div_sqrt50_fwd, eltw_add_bwd, eltw_add_fwd, eltw_mult_bwd, eltw_mult_fwd,
		layer_norm_bwd, layer_norm_fwd, matmul_bwd, matmul_fwd, softmax_y_bwd, softmax_y_fwd,
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

//TODO: optimize multi pass (multiple single head attention calculation) to single pass
//https://wikidocs.net/31379 multi head attention파트 보면 도움될듯? 거의 한번의 행렬곱으로 처리가능함.
pub fn build_4_head_attention8(
	g: &mut ComputationGraph,
	input: usize,
	wq: [usize; 4],
	wk: [usize; 4],
	wv: [usize; 4],
	wo: usize,
) -> usize {
	let mut heads = [0, 0, 0, 0];
	for i in 0..4 {
		let q = g.alloc();
		g.adj[q].op = (matmul_fwd, matmul_bwd);
		g.connect(wq[i], q);
		g.connect(input, q);

		let k = g.alloc();
		g.adj[k].op = (matmul_fwd, matmul_bwd);
		g.connect(wk[i], k);
		g.connect(input, k);

		let v = g.alloc();
		g.adj[v].op = (matmul_fwd, matmul_bwd);
		g.connect(wv[i], v);
		g.connect(input, v);

		let tp = g.alloc();
		g.adj[tp].op = (transpose_fwd, transpose_bwd);
		g.connect(k, tp);

		let kq = g.alloc();
		g.adj[kq].op = (matmul_fwd, matmul_bwd);
		g.connect(tp, kq);
		g.connect(q, kq);

		let div_hidden_sqrt = g.alloc();
		g.adj[div_hidden_sqrt].op = (div_sqrt2_fwd, div_sqrt2_bwd);
		g.connect(kq, div_hidden_sqrt);

		let attw = g.alloc();
		g.adj[attw].op = (softmax_y_fwd, softmax_y_bwd);
		g.connect(div_hidden_sqrt, attw);

		let atts = g.alloc();
		g.adj[atts].op = (matmul_fwd, matmul_bwd);
		g.connect(v, atts);
		g.connect(attw, atts);

		heads[i] = atts;
	}

	let concat = g.alloc();
	g.adj[concat].op = (concat4y_fwd, concat4y_bwd);
	g.connect(heads[0], concat);
	g.connect(heads[1], concat);
	g.connect(heads[2], concat);
	g.connect(heads[3], concat);

	let ret = g.alloc();
	g.adj[ret].op = (matmul_fwd, matmul_bwd);
	g.connect(wo, ret);
	g.connect(concat, ret);

	ret
}

pub fn build_4_head_attention200(
	g: &mut ComputationGraph,
	input: usize,
	wq: [usize; 4],
	wk: [usize; 4],
	wv: [usize; 4],
	wo: usize,
) -> usize {
	let mut heads = [0, 0, 0, 0];
	for i in 0..4 {
		let q = g.alloc();
		g.adj[q].op = (matmul_fwd, matmul_bwd);
		g.connect(wq[i], q);
		g.connect(input, q);

		let k = g.alloc();
		g.adj[k].op = (matmul_fwd, matmul_bwd);
		g.connect(wk[i], k);
		g.connect(input, k);

		let v = g.alloc();
		g.adj[v].op = (matmul_fwd, matmul_bwd);
		g.connect(wv[i], v);
		g.connect(input, v);

		let tp = g.alloc();
		g.adj[tp].op = (transpose_fwd, transpose_bwd);
		g.connect(k, tp);

		let kq = g.alloc();
		g.adj[kq].op = (matmul_fwd, matmul_bwd);
		g.connect(tp, kq);
		g.connect(q, kq);

		let div_hidden_sqrt = g.alloc();
		g.adj[div_hidden_sqrt].op = (div_sqrt50_fwd, div_sqrt50_bwd);
		g.connect(kq, div_hidden_sqrt);

		let attw = g.alloc();
		g.adj[attw].op = (softmax_y_fwd, softmax_y_bwd);
		g.connect(div_hidden_sqrt, attw);

		let atts = g.alloc();
		g.adj[atts].op = (matmul_fwd, matmul_bwd);
		g.connect(v, atts);
		g.connect(attw, atts);

		heads[i] = atts;
	}

	let concat = g.alloc();
	g.adj[concat].op = (concat4y_fwd, concat4y_bwd);
	g.connect(heads[0], concat);
	g.connect(heads[1], concat);
	g.connect(heads[2], concat);
	g.connect(heads[3], concat);

	let ret = g.alloc();
	g.adj[ret].op = (matmul_fwd, matmul_bwd);
	g.connect(wo, ret);
	g.connect(concat, ret);

	ret
}

pub fn build_encoder(
	g: &mut ComputationGraph,
	input: usize,
	wq: [usize; 4],
	wk: [usize; 4],
	wv: [usize; 4],
	wo: usize,
) -> usize {
	let att = build_4_head_attention8(g, input, wq, wk, wv, wo);

	let resi = g.alloc();
	g.adj[resi].op = (eltw_add_fwd, eltw_add_bwd);
	g.connect(input, resi);
	g.connect(att, resi);

	let ln = g.alloc();
	g.adj[ln].op = (layer_norm_fwd, layer_norm_bwd);
	g.connect(resi, ln);

	ln
}

pub fn build_gemm(g: &mut ComputationGraph, input: usize, weight: usize, bias: usize) -> usize {
	let mm = g.alloc();
	g.adj[mm].op = (matmul_fwd, matmul_bwd);
	g.connect(weight, mm);
	g.connect(input, mm);

	let add = g.alloc();
	g.adj[add].op = (eltw_add_fwd, eltw_add_bwd);
	g.connect(mm, add);
	g.connect(bias, add);

	add
}
