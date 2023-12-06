use std::{
	fs::{self, File},
	io::Read,
};

use dlrs::{
	computation_graph::ComputationGraph,
	graph_builder::build_4_head_attention,
	operation::{matmul_bwd, matmul_fwd, sigsum_bwd, sigsum_fwd},
};
use ndarray::Array4;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

extern crate serde;
extern crate serde_json;
use serde::{Deserialize, Serialize};
#[derive(Debug, Deserialize, Serialize)]
struct EssayData {
	paragraph: String,
	score: f32,
	prompt: String,
}

fn main() {
	let batch_size = 16;
	let learning_rate = 0.01;
	let mut rng = StdRng::seed_from_u64(987);

	println!("Reading data");
	let mut file = File::open("data/ko.vec").unwrap();

	// this code cause Error { kind: InvalidData, message: "stream did not contain valid UTF-8" }
	// let mut s = String::new();
	// file.read_to_string(&mut s).unwrap();
	let mut buf = vec![];
	file.read_to_end(&mut buf).unwrap();
	let s = String::from_utf8_lossy(&buf);
	let s = s.trim_start_matches("\u{feff}").to_owned(); //remove utf8 BOM

	println!("Parsing data");
	let mut it = s.split_whitespace();
	let asdf = it.next().unwrap();
	let n = asdf.parse::<usize>().unwrap();
	let m = it.next().unwrap().parse::<usize>().unwrap();
	let mut vocab = Vec::with_capacity(n);
	for i in 0..n {
		let word = it.next().unwrap();
		let mut vec = Vec::with_capacity(m);
		for j in 0..m {
			vec.push(it.next().unwrap().parse::<f32>().unwrap());
		}
		vocab.push((word.to_owned(), vec));
	}
	println!("Sorting data");
	vocab.sort_by_key(|(w, _)| w.clone());
	println!("DONE");

	let directory_path = "./data/essay/trainp";
	let dir_entries = fs::read_dir(directory_path).unwrap();
	let mut data_train = Vec::new();
	for entry in dir_entries {
		let entry = entry.unwrap();
		let file_path = entry.path();
		if file_path.is_file() {
			let mut file = File::open(&file_path).unwrap();
			let mut s = String::new();
			file.read_to_string(&mut s).unwrap();

			data_train.push(serde_json::from_str::<EssayData>(&s).unwrap());
		}
	}
	data_train.shuffle(&mut rng);
	let data_test = data_train
		.drain(0..data_train.len() / 100)
		.collect::<Vec<_>>();

	//Create Graph
	let mut g = ComputationGraph::new();
	let word_num = 2;
	let hidden_size = 8;

	let input = g.alloc();

	let wq1 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let mut wq1_data = [
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
	];

	let wk1 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let mut wk1_data = [
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
	];

	let wv1 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let mut wv1_data = [
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
	];
	let wo1 = g.alloc();
	let mut wo1_data = Array4::<f32>::from_shape_fn((1, 1, hidden_size, hidden_size), |_| {
		rng.gen_range(-1. ..=1.)
	})
	.permuted_axes([0, 1, 3, 2]);

	let rsqrt1 = g.alloc();
	let rsqrt1_data = Array4::<f32>::from_shape_vec(
		(1, 1, word_num, word_num),
		vec![
			1. / ((hidden_size / 4) as f32).sqrt(),
			1. / ((hidden_size / 4) as f32).sqrt(),
			1. / ((hidden_size / 4) as f32).sqrt(),
			1. / ((hidden_size / 4) as f32).sqrt(),
		],
	)
	.unwrap();

	let att1 = build_4_head_attention(&mut g, input, wq1, wk1, wv1, rsqrt1, wo1);

	let matmul1_weight = g.alloc();
	let mut matmul1_weight_data =
		Array4::<f32>::from_shape_fn((1, 1, hidden_size, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		});
	let matmul1 = g.alloc();
	g.adj[matmul1].op = (matmul_fwd, matmul_bwd);
	g.connect(matmul1_weight, matmul1);
	g.connect(att1, matmul1);

	let wq2 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let mut wq2_data = [
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
	];

	let wk2 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let mut wk2_data = [
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
	];

	let wv2 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let mut wv2_data = [
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		}),
	];
	let wo2 = g.alloc();
	let mut wo2_data = Array4::<f32>::from_shape_fn((1, 1, hidden_size, hidden_size), |_| {
		rng.gen_range(-1. ..=1.)
	})
	.permuted_axes([0, 1, 3, 2]);

	let rsqrt2 = g.alloc();
	let rsqrt2_data = Array4::<f32>::from_shape_vec(
		(1, 1, word_num, word_num),
		vec![
			1. / ((hidden_size / 4) as f32).sqrt(),
			1. / ((hidden_size / 4) as f32).sqrt(),
			1. / ((hidden_size / 4) as f32).sqrt(),
			1. / ((hidden_size / 4) as f32).sqrt(),
		],
	)
	.unwrap();

	let att2 = build_4_head_attention(&mut g, matmul1, wq2, wk2, wv2, rsqrt2, wo2);

	let matmul2_weight = g.alloc();
	let mut matmul2_weight_data =
		Array4::<f32>::from_shape_fn((1, 1, hidden_size, hidden_size), |_| {
			rng.gen_range(-1. ..=1.)
		});
	let matmul2 = g.alloc();
	g.adj[matmul2].op = (matmul_fwd, matmul_bwd);
	g.connect(matmul2_weight, matmul2);
	g.connect(att2, matmul2);

	let sigsum = g.alloc();
	g.adj[sigsum].op = (sigsum_fwd, sigsum_bwd);
	g.connect(matmul2, sigsum);

	for epoch in 0..100 {
		//TODO: input_data initialization
		//TODO: SGD
		let input_data = Array4::<f32>::from_shape_vec(
			(1, 1, word_num, hidden_size),
			vec![
				0.4657, 0.2328, 0.4527, 0.5871, 0.4086, 0.1272, 0.6373, 0.2421, 0.7312, 0.7224,
				0.1992, 0.6948, 0.5830, 0.6318, 0.5559, 0.1262,
			],
		)
		.unwrap()
		.permuted_axes([0, 1, 3, 2])
		.to_owned();
		let (res, grad) = g.run_essay(
			vec![
				(input, input_data.clone()),
				(wq1[0], wq1_data[0].clone()),
				(wq1[1], wq1_data[1].clone()),
				(wq1[2], wq1_data[2].clone()),
				(wq1[3], wq1_data[3].clone()),
				(wk1[0], wk1_data[0].clone()),
				(wk1[1], wk1_data[1].clone()),
				(wk1[2], wk1_data[2].clone()),
				(wk1[3], wk1_data[3].clone()),
				(wv1[0], wv1_data[0].clone()),
				(wv1[1], wv1_data[1].clone()),
				(wv1[2], wv1_data[2].clone()),
				(wv1[3], wv1_data[3].clone()),
				(wo1, wo1_data.clone()),
				(rsqrt1, rsqrt1_data.clone()),
				(matmul1_weight, matmul1_weight_data.clone()),
				(wq2[0], wq2_data[0].clone()),
				(wq2[1], wq2_data[1].clone()),
				(wq2[2], wq2_data[2].clone()),
				(wq2[3], wq2_data[3].clone()),
				(wk2[0], wk2_data[0].clone()),
				(wk2[1], wk2_data[1].clone()),
				(wk2[2], wk2_data[2].clone()),
				(wk2[3], wk2_data[3].clone()),
				(wv2[0], wv2_data[0].clone()),
				(wv2[1], wv2_data[1].clone()),
				(wv2[2], wv2_data[2].clone()),
				(wv2[3], wv2_data[3].clone()),
				(wo2, wo2_data.clone()),
				(rsqrt2, rsqrt2_data.clone()),
				(matmul2_weight, matmul2_weight_data.clone()),
			],
			1.,
		);
		// dbg!(&grad[matmul2_weight]);
		// dbg!(&grad[att2]);
		// dbg!(&grad[matmul2]);
		dbg!(&res[sigsum]);
		wq1_data[0] -= &(grad[wq1[0]].clone() * learning_rate);
		wq1_data[1] -= &(grad[wq1[1]].clone() * learning_rate);
		wq1_data[2] -= &(grad[wq1[2]].clone() * learning_rate);
		wq1_data[3] -= &(grad[wq1[3]].clone() * learning_rate);
		wk1_data[0] -= &(grad[wk1[0]].clone() * learning_rate);
		wk1_data[1] -= &(grad[wk1[1]].clone() * learning_rate);
		wk1_data[2] -= &(grad[wk1[2]].clone() * learning_rate);
		wk1_data[3] -= &(grad[wk1[3]].clone() * learning_rate);
		wv1_data[0] -= &(grad[wv1[0]].clone() * learning_rate);
		wv1_data[1] -= &(grad[wv1[1]].clone() * learning_rate);
		wv1_data[2] -= &(grad[wv1[2]].clone() * learning_rate);
		wv1_data[3] -= &(grad[wv1[3]].clone() * learning_rate);
		wo1_data -= &(grad[wo1].clone() * learning_rate);
		matmul1_weight_data -= &(grad[matmul1_weight].clone() * learning_rate);

		wq2_data[0] -= &(grad[wq2[0]].clone() * learning_rate);
		wq2_data[1] -= &(grad[wq2[1]].clone() * learning_rate);
		wq2_data[2] -= &(grad[wq2[2]].clone() * learning_rate);
		wq2_data[3] -= &(grad[wq2[3]].clone() * learning_rate);
		wk2_data[0] -= &(grad[wk2[0]].clone() * learning_rate);
		wk2_data[1] -= &(grad[wk2[1]].clone() * learning_rate);
		wk2_data[2] -= &(grad[wk2[2]].clone() * learning_rate);
		wk2_data[3] -= &(grad[wk2[3]].clone() * learning_rate);
		wv2_data[0] -= &(grad[wv2[0]].clone() * learning_rate);
		wv2_data[1] -= &(grad[wv2[1]].clone() * learning_rate);
		wv2_data[2] -= &(grad[wv2[2]].clone() * learning_rate);
		wv2_data[3] -= &(grad[wv2[3]].clone() * learning_rate);
		wo2_data -= &(grad[wo2].clone() * learning_rate);
		matmul2_weight_data -= &(grad[matmul2_weight].clone() * learning_rate);
	}
}
