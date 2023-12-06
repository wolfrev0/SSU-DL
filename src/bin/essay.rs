use std::{
	fs::{self, File},
	io::Read,
};

use dlrs::{computation_graph::ComputationGraph, graph_builder::build_4_head_attention};
use ndarray::Array4;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

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
	let input_data = Array4::<f32>::from_shape_vec(
		(1, 1, word_num, hidden_size),
		vec![
			0.4657, 0.2328, 0.4527, 0.5871, 0.4086, 0.1272, 0.6373, 0.2421, 0.7312, 0.7224, 0.1992,
			0.6948, 0.5830, 0.6318, 0.5559, 0.1262,
		],
	)
	.unwrap()
	.permuted_axes([0, 1, 3, 2])
	.to_owned();

	let wq1 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let wq1_data = [
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.0534, -0.3523, 0.2212, -0.2202, -0.0377, 0.2212, -0.0651, 0.3326, -0.1489,
				0.2738, 0.0497, 0.2621, 0.0573, -0.2022, 0.1499, 0.1341,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.2857, -0.3358, -0.4303, 0.4280, 0.0639, 0.0512, 0.0219, -0.1714, 0.3180, 0.2776,
				-0.3080, 0.1833, 0.2975, -0.0354, 0.0331, 0.0812,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.3593, -0.2785, 0.3906, -0.4104, 0.0785, -0.4186, -0.1351, -0.0531, 0.0308,
				-0.2086, 0.1734, 0.0921, -0.3620, 0.0373, 0.3828, 0.3626,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.1995, 0.2500, -0.3111, 0.3912, 0.1545, 0.1529, -0.3215, -0.1489, -0.3861,
				0.2099, -0.3645, -0.1652, -0.0355, 0.1767, 0.1136, 0.1304,
			],
		)
		.unwrap(),
	];

	let wk1 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let wk1_data = [
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.3395, 0.0103, -0.0497, -0.1656, -0.1073, -0.3533, 0.2709, 0.3934, 0.2670, 0.2093,
				-0.1221, -0.3563, 0.4111, -0.1301, -0.1681, -0.3245,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.2852, 0.0882, 0.2521, -0.3804, -0.3817, -0.0583, -0.3633, 0.3394, 0.0079,
				-0.3141, 0.1114, 0.3166, 0.0627, -0.3124, 0.0710, -0.3253,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.2476, -0.0419, 0.0978, -0.0579, -0.3474, 0.1200, 0.3684, -0.0865, -0.1574,
				-0.3525, -0.0513, 0.4203, -0.1165, -0.1281, 0.2959, -0.3529,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.0637, -0.1428, -0.0126, -0.2775, 0.0111, 0.0559, 0.0772, -0.2584, 0.1001,
				-0.3784, 0.0086, 0.0475, -0.1155, -0.0347, 0.0022, 0.2295,
			],
		)
		.unwrap(),
	];

	let wv1 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let wv1_data = [
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.3851, -0.3919, 0.3357, 0.1476, 0.4226, 0.3866, -0.1497, -0.2014, 0.2916, 0.2503,
				0.3599, -0.1842, 0.1081, -0.1516, -0.3214, 0.2700,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.0914, 0.3467, 0.2178, -0.3026, -0.2011, 0.1336, 0.3270, -0.3615, 0.4162, 0.2683,
				-0.0337, -0.4149, -0.1826, 0.2993, 0.1484, 0.0675,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.3655, -0.4087, -0.3786, 0.0012, -0.0587, -0.2419, 0.4113, -0.4243, -0.4085,
				-0.3800, -0.2139, 0.0300, -0.1254, 0.0323, 0.3303, -0.3276,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.1981, 0.4116, -0.0939, -0.1545, 0.2822, 0.3964, -0.1404, -0.3224, 0.4273,
				0.3636, 0.1640, 0.0624, -0.0505, -0.2415, -0.3907, 0.4238,
			],
		)
		.unwrap(),
	];
	let wo1 = g.alloc();
	let wo1_data = Array4::<f32>::from_shape_vec(
		(1, 1, hidden_size, hidden_size),
		vec![
			0.3387, 0.2434, -0.2648, -0.0385, 0.1132, -0.3144, -0.2423, 0.2218, 0.1567, -0.1614,
			-0.1412, 0.0777, 0.0554, 0.0766, -0.0468, 0.2696, -0.1261, -0.1694, -0.1721, -0.2212,
			0.1007, -0.2273, -0.2521, 0.1761, 0.1608, -0.2375, -0.1221, -0.2659, 0.0805, -0.0329,
			0.1880, -0.2263, -0.1175, 0.3200, 0.2771, 0.3436, 0.0953, 0.2695, 0.3105, -0.2706,
			-0.2586, 0.3115, 0.1275, 0.0393, 0.2626, -0.2982, 0.2530, 0.1796, 0.1200, 0.0578,
			-0.0828, 0.1529, 0.2779, 0.0422, -0.1554, -0.1785, -0.0185, -0.2612, -0.2105, 0.0981,
			-0.2829, 0.3263, 0.0247, 0.1514,
		],
	)
	.unwrap()
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

	let wq2 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let wq2_data = [
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.0534, -0.3523, 0.2212, -0.2202, -0.0377, 0.2212, -0.0651, 0.3326, -0.1489,
				0.2738, 0.0497, 0.2621, 0.0573, -0.2022, 0.1499, 0.1341,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.2857, -0.3358, -0.4303, 0.4280, 0.0639, 0.0512, 0.0219, -0.1714, 0.3180, 0.2776,
				-0.3080, 0.1833, 0.2975, -0.0354, 0.0331, 0.0812,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.3593, -0.2785, 0.3906, -0.4104, 0.0785, -0.4186, -0.1351, -0.0531, 0.0308,
				-0.2086, 0.1734, 0.0921, -0.3620, 0.0373, 0.3828, 0.3626,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.1995, 0.2500, -0.3111, 0.3912, 0.1545, 0.1529, -0.3215, -0.1489, -0.3861,
				0.2099, -0.3645, -0.1652, -0.0355, 0.1767, 0.1136, 0.1304,
			],
		)
		.unwrap(),
	];

	let wk2 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let wk2_data = [
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.3395, 0.0103, -0.0497, -0.1656, -0.1073, -0.3533, 0.2709, 0.3934, 0.2670, 0.2093,
				-0.1221, -0.3563, 0.4111, -0.1301, -0.1681, -0.3245,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.2852, 0.0882, 0.2521, -0.3804, -0.3817, -0.0583, -0.3633, 0.3394, 0.0079,
				-0.3141, 0.1114, 0.3166, 0.0627, -0.3124, 0.0710, -0.3253,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.2476, -0.0419, 0.0978, -0.0579, -0.3474, 0.1200, 0.3684, -0.0865, -0.1574,
				-0.3525, -0.0513, 0.4203, -0.1165, -0.1281, 0.2959, -0.3529,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.0637, -0.1428, -0.0126, -0.2775, 0.0111, 0.0559, 0.0772, -0.2584, 0.1001,
				-0.3784, 0.0086, 0.0475, -0.1155, -0.0347, 0.0022, 0.2295,
			],
		)
		.unwrap(),
	];

	let wv2 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let wv2_data = [
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.3851, -0.3919, 0.3357, 0.1476, 0.4226, 0.3866, -0.1497, -0.2014, 0.2916, 0.2503,
				0.3599, -0.1842, 0.1081, -0.1516, -0.3214, 0.2700,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.0914, 0.3467, 0.2178, -0.3026, -0.2011, 0.1336, 0.3270, -0.3615, 0.4162, 0.2683,
				-0.0337, -0.4149, -0.1826, 0.2993, 0.1484, 0.0675,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.3655, -0.4087, -0.3786, 0.0012, -0.0587, -0.2419, 0.4113, -0.4243, -0.4085,
				-0.3800, -0.2139, 0.0300, -0.1254, 0.0323, 0.3303, -0.3276,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.1981, 0.4116, -0.0939, -0.1545, 0.2822, 0.3964, -0.1404, -0.3224, 0.4273,
				0.3636, 0.1640, 0.0624, -0.0505, -0.2415, -0.3907, 0.4238,
			],
		)
		.unwrap(),
	];
	let wo2 = g.alloc();
	let wo2_data = Array4::<f32>::from_shape_vec(
		(1, 1, hidden_size, hidden_size),
		vec![
			0.3387, 0.2434, -0.2648, -0.0385, 0.1132, -0.3144, -0.2423, 0.2218, 0.1567, -0.1614,
			-0.1412, 0.0777, 0.0554, 0.0766, -0.0468, 0.2696, -0.1261, -0.1694, -0.1721, -0.2212,
			0.1007, -0.2273, -0.2521, 0.1761, 0.1608, -0.2375, -0.1221, -0.2659, 0.0805, -0.0329,
			0.1880, -0.2263, -0.1175, 0.3200, 0.2771, 0.3436, 0.0953, 0.2695, 0.3105, -0.2706,
			-0.2586, 0.3115, 0.1275, 0.0393, 0.2626, -0.2982, 0.2530, 0.1796, 0.1200, 0.0578,
			-0.0828, 0.1529, 0.2779, 0.0422, -0.1554, -0.1785, -0.0185, -0.2612, -0.2105, 0.0981,
			-0.2829, 0.3263, 0.0247, 0.1514,
		],
	)
	.unwrap()
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

	let att2 = build_4_head_attention(&mut g, att1, wq2, wk2, wv2, rsqrt2, wo2);

	for epoch in 0..10 {
		//TODO: Add MatMul and Softmax and Cross Entropy Loss
		//TODO: Change initial weights of second attention
		//TODO: input_data initialization
		//TODO: SGD
		let (res, grad) = g.run(vec![
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
		]);
		dbg!(&res[att2]);
	}
}
