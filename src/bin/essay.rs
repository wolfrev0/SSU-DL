use std::{
	fs::{self, File},
	io::Read,
};

use dlrs::{
	computation_graph::ComputationGraph,
	graph_builder::{build_encoder, build_gemm},
	operation::{relu_bwd, relu_fwd, sigmean_bwd, sigmean_fwd},
};
use ndarray::{Array4, Axis};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

extern crate serde;
extern crate serde_json;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Deserialize, Serialize)]
struct EssayData {
	paragraph: String,
	score: f32,
	prompt: String,
}

fn main() {
	let mut rng = StdRng::seed_from_u64(1333);

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
	for _ in 0..n {
		let word = it.next().unwrap();
		let mut vec = Vec::with_capacity(m);
		for _ in 0..m {
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
	let _ = data_test;

	//Create Graph
	let mut g = ComputationGraph::new();
	let hidden_size = 200;

	let input = g.alloc();

	let wq1 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let mut wq1_data = [
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
	];

	let wk1 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let mut wk1_data = [
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
	];

	let wv1 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let mut wv1_data = [
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
	];
	let wo1 = g.alloc();
	let mut wo1_data = Array4::<f32>::from_shape_fn((1, 1, hidden_size, hidden_size), |_| {
		rng.gen_range(-0.03..0.03)
	})
	.permuted_axes([0, 1, 3, 2]);
	let bo1 = g.alloc();
	let mut bo1_data =
		Array4::<f32>::from_shape_fn((1, 1, hidden_size, 1), |_| rng.gen_range(-0.03..0.03));

	let encoder1 = build_encoder(&mut g, input, wq1, wk1, wv1, wo1, bo1);

	let relu1a = g.alloc();
	g.adj[relu1a].op = (relu_fwd, relu_bwd);
	g.connect(encoder1, relu1a);

	let gemm1_weight = g.alloc();
	let mut gemm1_weight_data =
		Array4::<f32>::from_shape_fn((1, 1, hidden_size, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		});
	let gemm1_bias = g.alloc();
	let mut gemm1_bias_data =
		Array4::<f32>::from_shape_fn((1, 1, hidden_size, 1), |_| rng.gen_range(-0.03..0.03));
	let gemm1 = build_gemm(&mut g, relu1a, gemm1_weight, gemm1_bias);

	let relu1b = g.alloc();
	g.adj[relu1b].op = (relu_fwd, relu_bwd);
	g.connect(gemm1, relu1b);

	let wq2 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let mut wq2_data = [
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
	];

	let wk2 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let mut wk2_data = [
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
	];

	let wv2 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let mut wv2_data = [
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
		Array4::<f32>::from_shape_fn((1, 1, hidden_size / 4, hidden_size), |_| {
			rng.gen_range(-0.03..0.03)
		}),
	];
	let wo2 = g.alloc();
	let mut wo2_data = Array4::<f32>::from_shape_fn((1, 1, hidden_size, hidden_size), |_| {
		rng.gen_range(-0.03..0.03)
	});
	let bo2 = g.alloc();
	let mut bo2_data =
		Array4::<f32>::from_shape_fn((1, 1, hidden_size, 1), |_| rng.gen_range(-0.03..0.03));

	let encoder2 = build_encoder(&mut g, relu1b, wq2, wk2, wv2, wo2, bo2);

	let relu2a = g.alloc();
	g.adj[relu2a].op = (relu_fwd, relu_bwd);
	g.connect(encoder2, relu2a);

	let gemm2_weight = g.alloc();
	let mut gemm2_weight_data =
		Array4::<f32>::from_shape_fn((1, 1, 1, hidden_size), |_| rng.gen_range(-0.03..0.03));
	let gemm2_bias = g.alloc();
	let mut gemm2_bias_data =
		Array4::<f32>::from_shape_fn((1, 1, 1, 1), |_| rng.gen_range(-0.03..0.03));
	let gemm2 = build_gemm(&mut g, relu2a, gemm2_weight, gemm2_bias);

	let sigsum = g.alloc();
	g.adj[sigsum].op = (sigmean_fwd, sigmean_bwd);
	g.connect(gemm2, sigsum);
	let mut mse_sum = 0.;
	for data in data_test.clone() {
		let mut embvec = Vec::new();
		for line in data.paragraph.split('#') {
			for word in line.split('@') {
				let mut i = vocab.partition_point(|x| x.0.as_str() < word);
				if i == vocab.len() {
					i -= 1;
				}
				embvec.extend(vocab[i].1.clone().into_iter());
			}
		}
		let word_num = embvec.len() / hidden_size;
		let input_data = Array4::<f32>::from_shape_vec(
			(1, 1, word_num, hidden_size),
			embvec[..word_num * hidden_size].to_vec(),
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
				(bo1, bo1_data.clone()),
				(gemm1_weight, gemm1_weight_data.clone()),
				(gemm1_bias, gemm1_bias_data.clone()),
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
				(bo2, bo2_data.clone()),
				(gemm2_weight, gemm2_weight_data.clone()),
				(gemm2_bias, gemm2_bias_data.clone()),
			],
			data.score,
		);
		mse_sum += (res[sigsum].get((0, 0, 0, 0)).unwrap() - data.score).powi(2);
	}
	println!(
		"epoch {} avg test error: {}",
		0,
		mse_sum / data_test.len() as f32
	);
	for epoch in 0..10 {
		data_train.shuffle(&mut rng);
		for (data_idx, data) in data_train.clone().into_iter().enumerate() {
			let mut embvec = Vec::new();
			for line in data.paragraph.split('#') {
				for word in line.split('@') {
					let mut i = vocab.partition_point(|x| x.0.as_str() < word);
					if i == vocab.len() {
						i -= 1;
					}
					embvec.extend(vocab[i].1.clone().into_iter());
				}
			}
			let word_num = embvec.len() / hidden_size;
			let input_data = Array4::<f32>::from_shape_vec(
				(1, 1, word_num, hidden_size),
				embvec[..word_num * hidden_size].to_vec(),
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
					(bo1, bo1_data.clone()),
					(gemm1_weight, gemm1_weight_data.clone()),
					(gemm1_bias, gemm1_bias_data.clone()),
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
					(bo2, bo2_data.clone()),
					(gemm2_weight, gemm2_weight_data.clone()),
					(gemm2_bias, gemm2_bias_data.clone()),
				],
				data.score,
			);
			// dbg!(&res);
			// dbg!(&res[encoder2]);
			// dbg!(&grad[gemm2_weight]);
			// dbg!(&grad[encoder2]);
			// dbg!(&res[gemm2]);
			let learning_rate_base = 0.05;
			let learning_rate = learning_rate_base / (10.0 as f32).powi(epoch / 2);
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
			bo1_data -= &(grad[bo1].sum_axis(Axis(3)).insert_axis(Axis(3)) * learning_rate); //sum_axis required because it is broadcasted
			gemm1_weight_data -= &(grad[gemm1_weight].clone() * learning_rate);
			gemm1_bias_data -=
				&(grad[gemm1_bias].sum_axis(Axis(3)).insert_axis(Axis(3)) * learning_rate); //sum_axis required because it is broadcasted

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
			bo2_data -= &(grad[bo2].sum_axis(Axis(3)).insert_axis(Axis(3)) * learning_rate); //sum_axis required because it is broadcasted
			gemm2_weight_data -= &(grad[gemm2_weight].clone() * learning_rate);
			gemm2_bias_data -=
				&(grad[gemm2_bias].sum_axis(Axis(3)).insert_axis(Axis(3)) * learning_rate); //sum_axis required because it is broadcasted

			println!(
				"epoch {} {:.3}%, output: {:.3}, train error: {}",
				epoch + 1,
				data_idx as f32 / data_train.len() as f32 * 100.,
				res[sigsum].get((0, 0, 0, 0)).unwrap(),
				(res[sigsum].get((0, 0, 0, 0)).unwrap() - data.score).powi(2)
			);
		}

		let mut mse_sum = 0.;
		for data in data_test.clone() {
			let mut embvec = Vec::new();
			for line in data.paragraph.split('#') {
				for word in line.split('@') {
					let mut i = vocab.partition_point(|x| x.0.as_str() < word);
					if i == vocab.len() {
						i -= 1;
					}
					embvec.extend(vocab[i].1.clone().into_iter());
				}
			}
			let word_num = embvec.len() / hidden_size;
			let input_data = Array4::<f32>::from_shape_vec(
				(1, 1, word_num, hidden_size),
				embvec[..word_num * hidden_size].to_vec(),
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
					(bo1, bo1_data.clone()),
					(gemm1_weight, gemm1_weight_data.clone()),
					(gemm1_bias, gemm1_bias_data.clone()),
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
					(bo2, bo2_data.clone()),
					(gemm2_weight, gemm2_weight_data.clone()),
					(gemm2_bias, gemm2_bias_data.clone()),
				],
				data.score,
			);
			mse_sum += (res[sigsum].get((0, 0, 0, 0)).unwrap() - data.score).powi(2);
		}
		println!(
			"epoch {} avg test error: {}",
			epoch + 1,
			mse_sum / data_test.len() as f32
		);
	}
}
