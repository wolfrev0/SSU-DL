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

	let matmul1_weight = g.alloc();
	let matmul1_weight_data = Array4::<f32>::from_shape_vec(
		(1, 1, hidden_size, hidden_size),
		vec![
			-0.0056, -0.3306, -0.0718, 0.0073, -0.0990, 0.2854, -0.1452, -0.0226, -0.2679, 0.0129,
			-0.1377, -0.2226, 0.2605, 0.1444, -0.2826, -0.3319, 0.2197, 0.0772, 0.0358, 0.1623,
			0.0294, 0.2155, 0.1120, 0.2245, -0.2643, -0.3153, -0.1932, 0.2408, 0.0278, 0.3393,
			0.2543, 0.0691, 0.0944, 0.2078, 0.0266, 0.2149, 0.1077, -0.3328, 0.1626, 0.2379,
			-0.0853, 0.1442, -0.1377, 0.1454, -0.3254, 0.0798, -0.0648, 0.1908, 0.3174, -0.0453,
			0.0468, 0.0042, -0.0530, -0.1345, 0.0313, -0.2748, -0.2682, -0.1424, -0.1974, 0.1882,
			-0.0662, 0.3222, 0.1958, 0.0763,
		],
	)
	.unwrap();
	let matmul1 = g.alloc();
	g.adj[matmul1].op = (matmul_fwd, matmul_bwd);
	g.connect(matmul1_weight, matmul1);
	g.connect(att1, matmul1);

	let wq2 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let wq2_data = [
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.1746, -0.1138, 0.2381, -0.2624, 0.3319, -0.0357, -0.0685, 0.2264, 0.3388,
				-0.1218, -0.0653, 0.1610, -0.0349, -0.1885, -0.3245, -0.1809,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.0761, 0.2314, 0.3400, -0.1618, -0.0639, -0.1412, 0.1651, -0.2415, 0.3310,
				-0.3151, 0.1995, -0.0627, 0.1868, 0.2315, 0.3180, -0.1023,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.2352, 0.2508, 0.1143, 0.0588, -0.1865, 0.2999, 0.1388, 0.0185, -0.1901, 0.1015,
				-0.2660, 0.0820, -0.2918, -0.0786, -0.0823, -0.0904,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.3164, -0.3328, -0.0911, 0.0578, 0.2476, 0.3283, 0.0898, -0.3134, -0.3514,
				-0.1420, -0.1934, 0.0359, -0.1191, -0.2248, -0.2878, 0.0853,
			],
		)
		.unwrap(),
	];

	let wk2 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let wk2_data = [
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.2907, -0.1860, -0.3295, 0.3201, -0.2480, 0.1866, 0.0607, 0.2518, -0.1627, 0.1563,
				0.1882, 0.1876, -0.2763, 0.0922, 0.1852, -0.2950,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.1619, 0.0879, -0.0800, -0.0244, 0.0974, -0.3078, -0.1028, -0.2583, -0.1241,
				-0.2073, -0.1550, 0.2678, 0.2568, 0.2371, 0.2527, -0.2444,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.3150, -0.0027, -0.2780, 0.3112, 0.2409, 0.0756, 0.0952, -0.1718, -0.0016,
				0.0206, 0.0854, 0.0737, 0.3291, -0.1561, 0.1851, 0.0412,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.2921, 0.1733, 0.2655, 0.3446, 0.0735, -0.0717, -0.1568, 0.2549, -0.2516,
				-0.2206, 0.3149, -0.0183, 0.1801, 0.0430, -0.1162, 0.0427,
			],
		)
		.unwrap(),
	];

	let wv2 = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
	let wv2_data = [
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.1997, -0.0205, -0.0576, -0.0989, -0.3490, 0.0884, -0.1113, -0.2857, 0.2883,
				0.1691, 0.0094, -0.1861, -0.0550, 0.2209, -0.0960, 0.1730,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				-0.0090, 0.3071, -0.3329, 0.1058, -0.3442, 0.3250, 0.1452, 0.2723, 0.0837, 0.3498,
				0.3443, -0.0358, 0.2844, 0.0863, -0.2781, 0.1672,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.0826, 0.1951, 0.0811, 0.0215, -0.2072, 0.1274, -0.0934, 0.1191, 0.3409, 0.3192,
				0.0657, -0.0313, -0.0606, 0.2788, 0.1698, -0.0147,
			],
		)
		.unwrap(),
		Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size / 4, hidden_size),
			vec![
				0.3409, 0.3192, 0.0657, -0.0313, -0.0606, 0.2788, 0.1698, -0.0147, -0.0979,
				-0.1532, 0.0376, 0.0681, -0.2852, 0.1643, 0.2181, 0.3095,
			],
		)
		.unwrap(),
	];
	let wo2 = g.alloc();
	let wo2_data = Array4::<f32>::from_shape_vec(
		(1, 1, hidden_size, hidden_size),
		vec![
			-0.0056, -0.3306, -0.0718, 0.0073, -0.0990, 0.2854, -0.1452, -0.0226, -0.2679, 0.0129,
			-0.1377, -0.2226, 0.2605, 0.1444, -0.2826, -0.3319, 0.2197, 0.0772, 0.0358, 0.1623,
			0.0294, 0.2155, 0.1120, 0.2245, -0.2643, -0.3153, -0.1932, 0.2408, 0.0278, 0.3393,
			0.2543, 0.0691, 0.0944, 0.2078, 0.0266, 0.2149, 0.1077, -0.3328, 0.1626, 0.2379,
			-0.0853, 0.1442, -0.1377, 0.1454, -0.3254, 0.0798, -0.0648, 0.1908, 0.3174, -0.0453,
			0.0468, 0.0042, -0.0530, -0.1345, 0.0313, -0.2748, -0.2682, -0.1424, -0.1974, 0.1882,
			-0.0662, 0.3222, 0.1958, 0.0763,
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

	let att2 = build_4_head_attention(&mut g, matmul1, wq2, wk2, wv2, rsqrt2, wo2);

	let matmul2_weight = g.alloc();
	let matmul2_weight_data = Array4::<f32>::from_shape_vec(
		(1, 1, hidden_size, hidden_size),
		vec![
			-0.0056, -0.3306, -0.0718, 0.0073, -0.0990, 0.2854, -0.1452, -0.0226, -0.2679, 0.0129,
			-0.1377, -0.2226, 0.2605, 0.1444, -0.2826, -0.3319, 0.2197, 0.0772, 0.0358, 0.1623,
			0.0294, 0.2155, 0.1120, 0.2245, -0.2643, -0.3153, -0.1932, 0.2408, 0.0278, 0.3393,
			0.2543, 0.0691, 0.0944, 0.2078, 0.0266, 0.2149, 0.1077, -0.3328, 0.1626, 0.2379,
			-0.0853, 0.1442, -0.1377, 0.1454, -0.3254, 0.0798, -0.0648, 0.1908, 0.3174, -0.0453,
			0.0468, 0.0042, -0.0530, -0.1345, 0.0313, -0.2748, -0.2682, -0.1424, -0.1974, 0.1882,
			-0.0662, 0.3222, 0.1958, 0.0763,
		],
	)
	.unwrap();
	let matmul2 = g.alloc();
	g.adj[matmul2].op = (matmul_fwd, matmul_bwd);
	g.connect(matmul2_weight, matmul2);
	g.connect(att2, matmul2);

	let sigsum = g.alloc();
	g.adj[sigsum].op = (sigsum_fwd, sigsum_bwd);
	g.connect(matmul2, sigsum);

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
		]);
		dbg!(&res[sigsum]);
	}
}
