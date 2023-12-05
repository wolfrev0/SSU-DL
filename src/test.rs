#[cfg(test)]
mod tests {
	use ndarray::Array4;

	use crate::{
		computation_graph::ComputationGraph,
		graph_builder::{build_4_head_attention, build_attention},
		operation::{
			attention_bwd, attention_fwd, concat4x_bwd, concat4x_fwd, concat4y_bwd, concat4y_fwd,
			eltw_add_bwd, eltw_add_fwd, layer_norm_bwd, layer_norm_fwd, matmul_bwd, matmul_fwd,
			relu_bwd, relu_fwd, softmax_cross_entropy_bwd, softmax_cross_entropy_fwd,
		},
		util::is_equal,
	};

	#[test]
	fn test_relu() {
		let mut g = ComputationGraph::new();

		let input = g.alloc();
		let input_data =
			Array4::<f32>::from_shape_vec((1, 1, 2, 3), vec![0., 1., -2., 3., -4., 5.]).unwrap();

		let hidden = g.alloc();
		g.adj[hidden].op = (relu_fwd, relu_bwd);
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
		A = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
		B = torch.arange(3, dtype=torch.float32).reshape(3, 1).requires_grad_()
		C = torch.mm(A, B)
		loss = C.sum()
		loss.backward()
		grad_A = A.grad
		grad_B = B.grad
		print("Matrix A:")
		print(A)
		print("\nMatrix B:")
		print(B)
		print("\nMatrix C (result of A*B):")
		print(C)
		print("\nGradient of A:")
		print(grad_A)
		print("\nGradient of B:")
		print(grad_B)*/
		let mut g = ComputationGraph::new();

		let input = g.alloc();
		let input_data =
			Array4::<f32>::from_shape_vec((1, 1, 2, 3), vec![0., 1., 2., 3., 4., 5.]).unwrap();

		let weight1 = g.alloc();
		let weight1_data = Array4::<f32>::from_shape_vec((1, 1, 3, 1), vec![0., 1., 2.]).unwrap();

		let weight2 = g.alloc();
		let weight2_data = Array4::<f32>::from_shape_vec((1, 1, 1, 2), vec![1., 1.]).unwrap();

		let fc1 = g.alloc();
		g.adj[fc1].op = (matmul_fwd, matmul_bwd);
		g.connect(input, fc1);
		g.connect(weight1, fc1);

		let fc2 = g.alloc();
		g.adj[fc2].op = (matmul_fwd, matmul_bwd);
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
		torch.manual_seed(11111)
		class SimpleNet(nn.Module):
			def __init__(self, input_size, hidden_size, output_size):
				super(SimpleNet, self).__init__()
				self.fc1 = nn.Linear(input_size, hidden_size, False)
				self.relu1 = nn.ReLU()
				self.fc2 = nn.Linear(hidden_size, output_size, False)
				self.relu2 = nn.ReLU()

			def forward(self, x):
				x = self.relu1(self.fc1(x))
				x = self.relu2(self.fc2(x))
				return x

		input_size = 8
		hidden_size = 6
		output_size = 4

		model = SimpleNet(input_size, hidden_size, output_size)
		example_input = torch.rand((1, input_size))
		output = model(example_input)

		print(model)
		print("\nInput:")
		print(example_input)
		print("\nWeights of Linear Layer 1:")
		print(model.fc1.weight)
		print("\nWeights of Linear Layer 2:")
		print(model.fc2.weight)
		print("\nOutput:")
		print(output)

		output.backward(gradient=torch.tensor([[1.,1.,1.,1.]]))
		print("\nGradients of Linear Layer 1 weights:")
		print(model.fc1.weight.grad)
		print("\nGradients of Linear Layer 2 weights:")
		print(model.fc2.weight.grad)*/
		let mut g = ComputationGraph::new();

		let input = g.alloc();
		let input_data = Array4::<f32>::from_shape_vec(
			(1, 1, 8, 1),
			vec![
				0.7608, 0.9041, 0.9830, 0.3026, 0.3139, 0.1113, 0.2464, 0.9967,
			],
		)
		.unwrap();

		let weight1 = g.alloc();
		let weight1_data = Array4::<f32>::from_shape_vec(
			(1, 1, 6, 8),
			vec![
				0.1806, 0.0927, 0.0754, -0.1094, 0.0388, -0.2083, -0.3351, -0.2591, 0.2947,
				-0.2373, 0.2207, -0.2163, 0.0163, 0.1468, -0.3526, 0.3319, -0.2321, 0.1037,
				-0.2045, -0.1501, 0.2676, 0.3087, 0.3164, -0.0726, -0.1910, 0.2309, -0.2953,
				-0.0291, -0.1337, 0.2855, -0.0138, 0.0819, 0.0396, 0.1394, 0.2698, -0.1355,
				-0.1322, -0.2800, -0.3354, 0.1577, 0.3326, -0.1080, 0.0047, -0.1543, 0.0193,
				-0.0340, -0.3209, 0.0165,
			],
		)
		.unwrap();

		let weight2 = g.alloc();
		let weight2_data = Array4::<f32>::from_shape_vec(
			(1, 1, 4, 6),
			vec![
				0.1992, 0.1353, -0.3798, -0.1454, 0.3411, 0.0315, -0.3327, -0.1407, -0.1425,
				0.1139, -0.3173, 0.1386, 0.2275, -0.1266, -0.0111, 0.1349, 0.2610, -0.1336,
				-0.2452, 0.0849, 0.2613, 0.2177, -0.0315, -0.0740,
			],
		)
		.unwrap();

		let fc1 = g.alloc();
		g.adj[fc1].op = (matmul_fwd, matmul_bwd);
		g.connect(weight1, fc1);
		g.connect(input, fc1);

		let relu1 = g.alloc();
		g.adj[relu1].op = (relu_fwd, relu_bwd);
		g.connect(fc1, relu1);

		let fc2 = g.alloc();
		g.adj[fc2].op = (matmul_fwd, matmul_bwd);
		g.connect(weight2, fc2);
		g.connect(relu1, fc2);

		let relu2 = g.alloc();
		g.adj[relu2].op = (relu_fwd, relu_bwd);
		g.connect(fc2, relu2);

		let (res, grad) = g.run(vec![
			(input, input_data.clone()),
			(weight1, weight1_data.clone()),
			(weight2, weight2_data.clone()),
		]);
		// for i in grad.iter() {
		// 	println!("{}", i);
		// }
		assert!(is_equal(
			res[relu2].iter(),
			[0.1898, 0.0000, 0.0387, 0.0203].iter()
		));
		assert!(is_equal(
			grad[weight1].iter(),
			[
				0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0712, 0.0846,
				0.0920, 0.0283, 0.0294, 0.0104, 0.0231, 0.0933, 0.0000, 0.0000, 0.0000, 0.0000,
				0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
				0.0000, 0.0000, 0.4341, 0.5159, 0.5609, 0.1727, 0.1791, 0.0635, 0.1406, 0.5687,
				-0.1339, -0.1592, -0.1730, -0.0533, -0.0553, -0.0196, -0.0434, -0.1755
			]
			.iter()
		));
		assert!(is_equal(
			grad[weight2].iter(),
			[
				0.0000, 0.4265, 0.0000, 0.0000, 0.3823, 0.0529, 0.0000, 0.0000, 0.0000, 0.0000,
				0.0000, 0.0000, 0.0000, 0.4265, 0.0000, 0.0000, 0.3823, 0.0529, 0.0000, 0.4265,
				0.0000, 0.0000, 0.3823, 0.0529
			]
			.iter()
		));
	}

	#[test]
	fn softmax() {}
	#[test]
	fn cross_entropy() {}
	#[test]
	fn softmax_cross_entropy_test() {
		/*REFERENCE CODE
		import torch
		import torch.nn.functional as F
		torch.manual_seed(123)
		num_classes = 3
		logits = torch.randn(1, num_classes, requires_grad=True)
		target = torch.randint(0, num_classes, (1,))
		softmax_output = F.softmax(logits, dim=1)
		cross_entropy_loss = F.cross_entropy(logits, target)
		cross_entropy_loss.backward()
		gradients = logits.grad
		print("Logits:", logits)
		print("Softmax output:", softmax_output)
		print("Target:", target)
		print("Cross Entropy Loss:", cross_entropy_loss.item())
		print("Gradients:", gradients)*/
		let mut g = ComputationGraph::new();

		let input = g.alloc();
		let input_data =
			Array4::<f32>::from_shape_vec((1, 1, 3, 1), vec![-0.1115, 0.1204, -0.3696]).unwrap();

		let truth = g.alloc();
		let truth_data = Array4::<f32>::from_shape_vec((1, 1, 3, 1), vec![0., 0., 1.]).unwrap();

		let sc = g.alloc();
		g.adj[sc].op = (softmax_cross_entropy_fwd, softmax_cross_entropy_bwd);
		g.connect(input, sc);
		g.connect(truth, sc);

		let (res, grad) = g.run(vec![
			(input, input_data.clone()),
			(truth, truth_data.clone()),
		]);
		// for i in res.iter() {
		// 	println!("{}", i);
		// }
		// for i in grad.iter() {
		// 	println!("{}", i);
		// }
		assert!(is_equal(res[sc].iter(), [1.3678419589996338].iter()));
		assert!(is_equal(
			grad[input].iter(),
			[0.3297, 0.4157, -0.7453].iter()
		));
	}
	#[test]
	fn fc_relu_resi_fc_relu() {
		/*REFERENCE CODE
		import torch
		import torch.nn as nn
		torch.manual_seed(11111)
		class SimpleNet(nn.Module):
			def __init__(self, input_size, hidden_size, output_size):
				super(SimpleNet, self).__init__()
				self.fc1 = nn.Linear(input_size, hidden_size, False)
				self.relu1 = nn.ReLU()
				self.fc2 = nn.Linear(hidden_size, output_size, False)
				self.relu2 = nn.ReLU()

			def forward(self, x):
				x_resi = x
				x = self.relu1(self.fc1(x))
				x = x + x_resi #residual connection
				x = self.relu2(self.fc2(x))
				return x

		input_size = 4
		hidden_size = 4
		output_size = 2

		model = SimpleNet(input_size, hidden_size, output_size)
		example_input = torch.rand((1, input_size))
		output = model(example_input)

		print(model)
		print("\nInput:")
		print(example_input)
		print("\nWeights of Linear Layer 1:")
		print(model.fc1.weight)
		print("\nWeights of Linear Layer 2:")
		print(model.fc2.weight)
		print("\nOutput:")
		print(output)

		output.backward(gradient=torch.tensor([[1.,1.]]))
		print("\nGradients of Linear Layer 1 weights:")
		print(model.fc1.weight.grad)
		print("\nGradients of Linear Layer 2 weights:")
		print(model.fc2.weight.grad)*/
		let mut g = ComputationGraph::new();

		let input = g.alloc();
		let input_data =
			Array4::<f32>::from_shape_vec((1, 1, 4, 1), vec![0.2300, 0.8265, 0.0824, 0.4588])
				.unwrap();

		let weight1 = g.alloc();
		let weight1_data = Array4::<f32>::from_shape_vec(
			(1, 1, 4, 4),
			vec![
				0.2554, 0.1311, 0.1066, -0.1548, 0.0549, -0.2946, -0.4739, -0.3664, 0.4168,
				-0.3356, 0.3121, -0.3059, 0.0231, 0.2076, -0.4987, 0.4694,
			],
		)
		.unwrap();

		let weight2 = g.alloc();
		let weight2_data = Array4::<f32>::from_shape_vec(
			(1, 1, 2, 4),
			vec![
				-0.3282, 0.1467, -0.2893, -0.2123, 0.3784, 0.4366, 0.4475, -0.1026,
			],
		)
		.unwrap();

		let fc1 = g.alloc();
		g.adj[fc1].op = (matmul_fwd, matmul_bwd);
		g.connect(weight1, fc1);
		g.connect(input, fc1);

		let relu1 = g.alloc();
		g.adj[relu1].op = (relu_fwd, relu_bwd);
		g.connect(fc1, relu1);

		let resi = g.alloc();
		g.adj[resi].op = (eltw_add_fwd, eltw_add_bwd);
		g.connect(input, resi);
		g.connect(relu1, resi);

		let fc2 = g.alloc();
		g.adj[fc2].op = (matmul_fwd, matmul_bwd);
		g.connect(weight2, fc2);
		g.connect(resi, fc2);

		let relu2 = g.alloc();
		g.adj[relu2].op = (relu_fwd, relu_bwd);
		g.connect(fc2, relu2);

		let (res, grad) = g.run(vec![
			(input, input_data.clone()),
			(weight1, weight1_data.clone()),
			(weight2, weight2_data.clone()),
		]);
		// for i in grad.iter() {
		// 	println!("{}", i);
		// }
		assert!(is_equal(res[relu2].iter(), [0.0000, 0.4413].iter()));
		assert!(is_equal(
			grad[weight1].iter(),
			[
				0.0870, 0.3128, 0.0312, 0.1736, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
				0.0000, 0.0000, -0.0236, -0.0848, -0.0085, -0.0471
			]
			.iter()
		));
		assert!(is_equal(
			grad[weight2].iter(),
			[0.0000, 0.0000, 0.0000, 0.0000, 0.3348, 0.8265, 0.0824, 0.8100].iter()
		));
	}
	#[test]
	fn fc_relu_fc_relu_resi() {
		/*REFERENCE CODE
		import torch
		import torch.nn as nn
		torch.manual_seed(11111)
		class SimpleNet(nn.Module):
			def __init__(self, input_size, hidden_size, output_size):
				super(SimpleNet, self).__init__()
				self.fc1 = nn.Linear(input_size, hidden_size, False)
				self.relu1 = nn.ReLU()
				self.fc2 = nn.Linear(hidden_size, output_size, False)
				self.relu2 = nn.ReLU()

			def forward(self, x):
				x = self.relu1(self.fc1(x))
				x_resi = x
				x = self.relu2(self.fc2(x))
				x = x + x_resi #residual connection
				return x

		input_size = 4
		hidden_size = 3
		output_size = 3

		model = SimpleNet(input_size, hidden_size, output_size)
		example_input = torch.rand((1, input_size))
		output = model(example_input)

		print(model)
		print("\nInput:")
		print(example_input)
		print("\nWeights of Linear Layer 1:")
		print(model.fc1.weight)
		print("\nWeights of Linear Layer 2:")
		print(model.fc2.weight)
		print("\nOutput:")
		print(output)

		output.backward(gradient=torch.tensor([[1.,1.,1.]]))
		print("\nGradients of Linear Layer 1 weights:")
		print(model.fc1.weight.grad)
		print("\nGradients of Linear Layer 2 weights:")
		print(model.fc2.weight.grad)*/
		let mut g = ComputationGraph::new();

		let input = g.alloc();
		let input_data =
			Array4::<f32>::from_shape_vec((1, 1, 4, 1), vec![0.9366, 0.9475, 0.3974, 0.2300])
				.unwrap();

		let weight1 = g.alloc();
		let weight1_data = Array4::<f32>::from_shape_vec(
			(1, 1, 3, 4),
			vec![
				0.2554, 0.1311, 0.1066, -0.1548, 0.0549, -0.2946, -0.4739, -0.3664, 0.4168,
				-0.3356, 0.3121, -0.3059,
			],
		)
		.unwrap();

		let weight2 = g.alloc();
		let weight2_data = Array4::<f32>::from_shape_vec(
			(1, 1, 3, 3),
			vec![
				0.0266, 0.2398, -0.5758, 0.5420, -0.3790, 0.1694, -0.3340, -0.2452, 0.4370,
			],
		)
		.unwrap();

		let fc1 = g.alloc();
		g.adj[fc1].op = (matmul_fwd, matmul_bwd);
		g.connect(weight1, fc1);
		g.connect(input, fc1);

		let relu1 = g.alloc();
		g.adj[relu1].op = (relu_fwd, relu_bwd);
		g.connect(fc1, relu1);

		let fc2 = g.alloc();
		g.adj[fc2].op = (matmul_fwd, matmul_bwd);
		g.connect(weight2, fc2);
		g.connect(relu1, fc2);

		let relu2 = g.alloc();
		g.adj[relu2].op = (relu_fwd, relu_bwd);
		g.connect(fc2, relu2);

		let resi = g.alloc();
		g.adj[resi].op = (eltw_add_fwd, eltw_add_bwd);
		g.connect(relu1, resi);
		g.connect(relu2, resi);

		let (res, grad) = g.run(vec![
			(input, input_data.clone()),
			(weight1, weight1_data.clone()),
			(weight2, weight2_data.clone()),
		]);
		// for i in grad.iter() {
		// 	println!("{}", i);
		// }
		assert!(is_equal(res[resi].iter(), [0.3702, 0.2220, 0.1260].iter()));
		assert!(is_equal(
			grad[weight1].iter(),
			[
				1.4442, 1.4611, 0.6127, 0.3546, 0.0000, 0.0000, 0.0000, 0.0000, 1.0952, 1.1080,
				0.4647, 0.2689
			]
			.iter()
		));
		assert!(is_equal(
			grad[weight2].iter(),
			[0.0000, 0.0000, 0.0000, 0.3702, 0.0000, 0.1260, 0.0000, 0.0000, 0.0000].iter()
		));
	}
	#[test]
	fn attention_fwd_test() {
		/*REFERENCE CODE
		import torch
		import torch.nn as nn
		import torch.nn.functional as F

		torch.manual_seed(12)

		class SelfAttention(nn.Module):
			def __init__(self, hidden_size):
				super(SelfAttention, self).__init__()
				self.hidden_size = hidden_size
				self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
				self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
				self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)

			def forward(self, x):
				Q = self.W_q(x)
				K = self.W_k(x)
				V = self.W_v(x)
				scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
				attention_weights = F.softmax(scores, dim=-1)
				attended_values = torch.matmul(attention_weights, V)
				return attended_values

		# Test the SelfAttention module
		hidden_size = 4

		example_input = torch.rand((3, hidden_size))
		attention_module = SelfAttention(hidden_size)
		output = attention_module(example_input)

		print(attention_module)
		print("\nInput:")
		print(example_input)
		print("\nW_q:")
		print(attention_module.W_q.weight)
		print("\nW_k:")
		print(attention_module.W_k.weight)
		print("\nW_v:")
		print(attention_module.W_v.weight)
		print("\nAttention Module Weights:")
		print(attention_module.parameters())
		print("\nOutput:")
		print(output)*/
		let mut g = ComputationGraph::new();

		let input = g.alloc();
		let input_data = Array4::<f32>::from_shape_vec(
			(1, 1, 4, 3),
			vec![
				0.4657, 0.4086, 0.7312, 0.2328, 0.1272, 0.7224, 0.4527, 0.6373, 0.1992, 0.5871,
				0.2421, 0.6948,
			],
		)
		.unwrap();

		let wq = g.alloc();
		let wq_data = Array4::<f32>::from_shape_vec(
			(1, 1, 4, 4),
			vec![
				0.0830, 0.1318, 0.0559, -0.3738, 0.4790, 0.3443, -0.3744, -0.0544, 0.1601, -0.4446,
				-0.3427, 0.3137, 0.2216, -0.2283, -0.1997, 0.1099,
			],
		)
		.unwrap();

		let wk = g.alloc();
		let wk_data = Array4::<f32>::from_shape_vec(
			(1, 1, 4, 4),
			vec![
				0.0784, 0.1083, -0.0661, 0.3813, -0.1784, -0.2396, -0.2434, -0.3128, 0.1423,
				-0.3214, -0.3565, 0.2490, 0.2275, -0.3359, -0.1727, -0.3761,
			],
		)
		.unwrap();

		let wv = g.alloc();
		let wv_data = Array4::<f32>::from_shape_vec(
			(1, 1, 4, 4),
			vec![
				0.1138, -0.0465, 0.2659, -0.3200, -0.1662, 0.4526, 0.3919, 0.4859, 0.1348, 0.3811,
				0.4391, -0.3827, -0.3658, 0.4405, 0.1803, 0.0556,
			],
		)
		.unwrap();

		let att = g.alloc();
		g.adj[att].op = (attention_fwd, attention_bwd);
		g.connect(input, att);
		g.connect(wq, att);
		g.connect(wk, att);
		g.connect(wv, att);

		let (res, grad) = g.run(vec![
			(input, input_data.clone()),
			(wq, wq_data.clone()),
			(wk, wk_data.clone()),
			(wv, wv_data.clone()),
		]);
		// for i in res.iter() {
		// 	println!("{}", i);
		// }
		// for i in grad.iter() {
		// 	println!("{}", i);
		// }
		assert!(is_equal(
			res[att].iter(),
			[
				-0.0028, -0.0040, -0.0009, 0.4882, 0.4895, 0.4862, 0.2045, 0.2041, 0.2052, 0.0684,
				0.0689, 0.0677
			]
			.iter()
		));
		// todo!("gradient assertion");
		// assert!(is_equal(grad[wq].iter(), [].iter()));
	}
	#[test]
	fn attention_manual_test() {
		/*REFERENCE CODE
		import torch
		import torch.nn as nn
		import torch.nn.functional as F

		torch.manual_seed(12)

		class SelfAttention(nn.Module):
			def __init__(self, hidden_size):
				super(SelfAttention, self).__init__()
				self.hidden_size = hidden_size
				self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
				self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
				self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)

			def forward(self, x):
				Q = self.W_q(x)
				K = self.W_k(x)
				V = self.W_v(x)
				scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
				attention_weights = F.softmax(scores, dim=-1)
				attended_values = torch.matmul(attention_weights, V)
				return attended_values

		# Test the SelfAttention module
		hidden_size = 4

		example_input = torch.rand((3, hidden_size))
		attention_module = SelfAttention(hidden_size)
		output = attention_module(example_input)

		print(attention_module)
		print("\nInput:")
		print(example_input)
		print("\nW_q:")
		print(attention_module.W_q.weight)
		print("\nW_k:")
		print(attention_module.W_k.weight)
		print("\nW_v:")
		print(attention_module.W_v.weight)
		print("\nAttention Module Weights:")
		print(attention_module.parameters())
		print("\nOutput:")
		print(output)*/
		let mut g = ComputationGraph::new();
		let word_num = 3;
		let hidden_size = 4;

		let input = g.alloc();
		let input_data = Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size, word_num),
			vec![
				0.4657, 0.4086, 0.7312, 0.2328, 0.1272, 0.7224, 0.4527, 0.6373, 0.1992, 0.5871,
				0.2421, 0.6948,
			],
		)
		.unwrap();

		let wq = g.alloc();
		let wq_data = Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size, hidden_size),
			vec![
				0.0830, 0.1318, 0.0559, -0.3738, 0.4790, 0.3443, -0.3744, -0.0544, 0.1601, -0.4446,
				-0.3427, 0.3137, 0.2216, -0.2283, -0.1997, 0.1099,
			],
		)
		.unwrap();

		let wk = g.alloc();
		let wk_data = Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size, hidden_size),
			vec![
				0.0784, 0.1083, -0.0661, 0.3813, -0.1784, -0.2396, -0.2434, -0.3128, 0.1423,
				-0.3214, -0.3565, 0.2490, 0.2275, -0.3359, -0.1727, -0.3761,
			],
		)
		.unwrap();

		let wv = g.alloc();
		let wv_data = Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size, hidden_size),
			vec![
				0.1138, -0.0465, 0.2659, -0.3200, -0.1662, 0.4526, 0.3919, 0.4859, 0.1348, 0.3811,
				0.4391, -0.3827, -0.3658, 0.4405, 0.1803, 0.0556,
			],
		)
		.unwrap();

		let rsqrt = g.alloc();
		let rsqrt_data = Array4::<f32>::from_shape_vec(
			(1, 1, word_num, word_num),
			vec![
				1. / (hidden_size as f32).sqrt(),
				1. / (hidden_size as f32).sqrt(),
				1. / (hidden_size as f32).sqrt(),
				1. / (hidden_size as f32).sqrt(),
				1. / (hidden_size as f32).sqrt(),
				1. / (hidden_size as f32).sqrt(),
				1. / (hidden_size as f32).sqrt(),
				1. / (hidden_size as f32).sqrt(),
				1. / (hidden_size as f32).sqrt(),
			],
		)
		.unwrap();

		let atts = build_attention(&mut g, input, wq, wk, wv, rsqrt);

		let (res, grad) = g.run(vec![
			(input, input_data.clone()),
			(wq, wq_data.clone()),
			(wk, wk_data.clone()),
			(wv, wv_data.clone()),
			(rsqrt, rsqrt_data.clone()),
		]);
		// for i in res.iter() {
		// 	println!("{}", i);
		// }
		// for i in grad.iter() {
		// 	println!("{}", i);
		// }
		assert!(is_equal(
			res[atts].iter(),
			[
				-0.0028, -0.0040, -0.0009, 0.4882, 0.4895, 0.4862, 0.2045, 0.2041, 0.2052, 0.0684,
				0.0689, 0.0677
			]
			.iter()
		));
		assert!(is_equal(
			grad[wq].iter(),
			[
				-0.0003, -0.0002, -0.0002, -0.0003, -0.0007, -0.0005, -0.0006, -0.0007, -0.0020,
				-0.0014, -0.0016, -0.0019, 0.0002, 0.0001, 0.0001, 0.0002
			]
			.iter()
		));
		assert!(is_equal(
			grad[wk].iter(),
			[
				-4.8840e-04,
				-8.9993e-04,
				1.8532e-04,
				4.9707e-04,
				1.2508e-03,
				2.3046e-03,
				-4.5585e-04,
				-1.3193e-03,
				-4.2229e-04,
				-7.7814e-04,
				1.6816e-04,
				4.1017e-04,
				3.7838e-05,
				6.9710e-05,
				-1.0619e-05,
				-4.7758e-05
			]
			.iter()
		));
		assert!(is_equal(
			grad[wv].iter(),
			[
				1.5995, 1.0712, 1.2975, 1.5153, 1.5995, 1.0712, 1.2975, 1.5153, 1.5995, 1.0712,
				1.2975, 1.5153, 1.5995, 1.0712, 1.2975, 1.5153
			]
			.iter()
		));
	}

	#[test]
	fn test_layer_normalization() {
		/*REFERENCE CODE
		import torch
		import torch.nn as nn
		from torch.autograd import Variable

		class SimpleNN(nn.Module):
			def __init__(self, input_size):
				super(SimpleNN, self).__init__()
				self.layer_norm = nn.LayerNorm(input_size)

			def forward(self, x):
				x = self.layer_norm(x)
				self.asdf=x
				self.asdf.retain_grad()
				x = nn.ReLU(False)(x)

				return x

		input_size = 4
		batch_size = 3
		model = SimpleNN(input_size)

		input_data = torch.tensor([5,1,0,0,0,1,1,1,0,1,0,9,], dtype=torch.float32).view((batch_size,input_size))
		input_data = Variable(input_data, requires_grad=True)

		print("Input:")
		print(input_data)

		print("\nGamma:")
		print(model.layer_norm.weight)

		print("\nBeta:")
		print(model.layer_norm.bias)

		output = model(input_data)

		print("\nOutput:")
		print(output)

		output.sum().backward()

		print("\nReLU Gradient:")
		print(model.asdf.grad)

		print("\nInput Gradient:")
		print(input_data.grad)*/
		let mut g = ComputationGraph::new();

		let input = g.alloc();
		let input_data = Array4::<f32>::from_shape_vec(
			(3, 1, 1, 4),
			vec![5., 1., 0., 0., 0., 1., 1., 1., 0., 1., 0., 9.],
		)
		.unwrap();

		let ln = g.alloc();
		g.adj[ln].op = (layer_norm_fwd, layer_norm_bwd);
		g.connect(input, ln);

		let relu = g.alloc();
		g.adj[relu].op = (relu_fwd, relu_bwd);
		g.connect(ln, relu);

		let (out, grad) = g.run(vec![(input, input_data.clone())]);
		assert!(is_equal(
			out[relu].iter(),
			[
				1.6977, 0.0000, 0.0000, 0.0000, 0.0000, 0.5773, 0.5773, 0.5773, 0.0000, 0.0000,
				0.0000, 1.7219
			]
			.iter()
		));
		assert!(is_equal(
			grad[input].iter(),
			[
				1.4268e-02,
				-7.1334e-02,
				2.8533e-02,
				2.8533e-02,
				-9.2268e-05,
				3.0756e-05,
				3.0756e-05,
				3.0756e-05,
				9.2949e-03,
				-2.0914e-02,
				9.2949e-03,
				2.3239e-03
			]
			.iter()
		));
	}

	#[test]
	fn test_concat4x() {
		let mut g = ComputationGraph::new();

		let m0 = g.alloc();
		let m0dat = Array4::<f32>::from_shape_vec(
			(2, 2, 2, 2),
			vec![
				0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 1., 2., 3., 4., 5.,
			],
		)
		.unwrap();

		let m1 = g.alloc();
		let m1dat = Array4::<f32>::from_shape_vec(
			(2, 2, 2, 2),
			vec![
				1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 1., 2., 3., 4., 5., 6.,
			],
		)
		.unwrap();

		let m2 = g.alloc();
		let m2dat = Array4::<f32>::from_shape_vec(
			(2, 2, 2, 2),
			vec![
				2., 3., 4., 5., 6., 7., 8., 9., 0., 1., 2., 3., 4., 5., 6., 7.,
			],
		)
		.unwrap();

		let m3 = g.alloc();
		let m3dat = Array4::<f32>::from_shape_vec(
			(2, 2, 2, 2),
			vec![
				3., 4., 5., 6., 7., 8., 9., 0., 1., 2., 3., 4., 5., 6., 7., 8.,
			],
		)
		.unwrap();

		let concat = g.alloc();
		g.adj[concat].op = (concat4x_fwd, concat4x_bwd);
		g.connect(m0, concat);
		g.connect(m1, concat);
		g.connect(m2, concat);
		g.connect(m3, concat);

		let (out, grad) = g.run(vec![
			(m0, m0dat.clone()),
			(m1, m1dat.clone()),
			(m2, m2dat.clone()),
			(m3, m3dat.clone()),
		]);
		// dbg!(&out[concat]);
		// dbg!(&grad[m0]);
		// dbg!(&grad[m1]);
		// dbg!(&grad[m2]);
		// dbg!(&grad[m3]);

		//assertion: TODO
	}

	#[test]
	fn test_concat4y() {
		let mut g = ComputationGraph::new();

		let m0 = g.alloc();
		let m0dat = Array4::<f32>::from_shape_vec(
			(2, 2, 2, 2),
			vec![
				0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 1., 2., 3., 4., 5.,
			],
		)
		.unwrap();

		let m1 = g.alloc();
		let m1dat = Array4::<f32>::from_shape_vec(
			(2, 2, 2, 2),
			vec![
				1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 1., 2., 3., 4., 5., 6.,
			],
		)
		.unwrap();

		let m2 = g.alloc();
		let m2dat = Array4::<f32>::from_shape_vec(
			(2, 2, 2, 2),
			vec![
				2., 3., 4., 5., 6., 7., 8., 9., 0., 1., 2., 3., 4., 5., 6., 7.,
			],
		)
		.unwrap();

		let m3 = g.alloc();
		let m3dat = Array4::<f32>::from_shape_vec(
			(2, 2, 2, 2),
			vec![
				3., 4., 5., 6., 7., 8., 9., 0., 1., 2., 3., 4., 5., 6., 7., 8.,
			],
		)
		.unwrap();

		let concat = g.alloc();
		g.adj[concat].op = (concat4y_fwd, concat4y_bwd);
		g.connect(m0, concat);
		g.connect(m1, concat);
		g.connect(m2, concat);
		g.connect(m3, concat);

		let (out, grad) = g.run(vec![
			(m0, m0dat.clone()),
			(m1, m1dat.clone()),
			(m2, m2dat.clone()),
			(m3, m3dat.clone()),
		]);
		dbg!(&out[concat]);
		dbg!(&grad[m0]);
		dbg!(&grad[m1]);
		dbg!(&grad[m2]);
		dbg!(&grad[m3]);

		//assertion: TODO
	}

	#[test]
	fn test_mha() {
		/*REFERENCE CODE
		import torch
		import torch.nn.functional as F
		import torch.nn as nn

		torch.manual_seed(12)

		class MultiheadAttention(nn.Module):
				def __init__(self, input_size, num_heads, dropout=0.1):
						super(MultiheadAttention, self).__init__()

						self.input_size = input_size
						self.num_heads = num_heads
						self.head_dim = input_size // num_heads

						self.W_q = nn.Linear(input_size, input_size, bias=False)
						self.W_k = nn.Linear(input_size, input_size, bias=False)
						self.W_v = nn.Linear(input_size, input_size, bias=False)
						self.W_o = nn.Linear(input_size, input_size, bias=False)

						# # Dropout layer
						# self.dropout = nn.Dropout(dropout)

				def forward(self, query, key, value, mask=None):
						batch_size = query.size(0)
						Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim)
						K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim)
						V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim)
						Q = Q.transpose(1, 2)
						K = K.transpose(1, 2)
						V = V.transpose(1, 2)
						scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
						if mask is not None:
								scores = scores.masked_fill(mask == 0, float("-inf"))
						attention_weights = F.softmax(scores, dim=-1)
						attention_output = torch.matmul(attention_weights, V)
						attention_output = attention_output.transpose(1, 2).contiguous()
						attention_output = attention_output.view(batch_size, -1, self.input_size)
						output = self.W_o(attention_output)
						return output
		input_size = 8
		num_heads = 4
		seq_length = 2
		batch_size = 1
		input = torch.randn(batch_size, seq_length, input_size)
		mha = MultiheadAttention(input_size, num_heads)
		output = mha(input, input, input)
		output.backward(gradient=torch.tensor([[[1.,1.,1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.,1.,1.]]]));
		print("Input:", input)
		print("Output:", output)
		print("Grad Wq:", mha.W_q.weight.grad)
		print("Grad Wk:", mha.W_k.weight.grad)
		print("Grad Wv:", mha.W_v.weight.grad)*/

		let mut g = ComputationGraph::new();
		let word_num = 2;
		let hidden_size = 8;

		let input = g.alloc();
		let input_data = Array4::<f32>::from_shape_vec(
			(1, 1, word_num, hidden_size),
			vec![
				-0.1320, -0.1254, 0.3443, -0.4519, -0.8888, -0.3526, -1.3373, 0.5223, -1.1118,
				-0.7171, 1.0426, -1.2510, -0.5107, -0.3843, -0.4899, 0.5306,
			],
		)
		.unwrap()
		.permuted_axes([0, 1, 3, 2])
		.to_owned();

		let wq = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
		let wq_data = [
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					0.3387, 0.2434, -0.2648, -0.0385, 0.1132, -0.3144, -0.2423, 0.2218, 0.1567,
					-0.1614, -0.1412, 0.0777, 0.0554, 0.0766, -0.0468, 0.2696,
				],
			)
			.unwrap(),
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					-0.1261, -0.1694, -0.1721, -0.2212, 0.1007, -0.2273, -0.2521, 0.1761, 0.1608,
					-0.2375, -0.1221, -0.2659, 0.0805, -0.0329, 0.1880, -0.2263,
				],
			)
			.unwrap(),
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					-0.1175, 0.3200, 0.2771, 0.3436, 0.0953, 0.2695, 0.3105, -0.2706, -0.2586,
					0.3115, 0.1275, 0.0393, 0.2626, -0.2982, 0.2530, 0.1796,
				],
			)
			.unwrap(),
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					0.1200, 0.0578, -0.0828, 0.1529, 0.2779, 0.0422, -0.1554, -0.1785, -0.0185,
					-0.2612, -0.2105, 0.0981, -0.2829, 0.3263, 0.0247, 0.1514,
				],
			)
			.unwrap(),
		];

		let wk = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
		let wk_data = [
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					0.0436, -0.2877, 0.1806, -0.1798, -0.0308, 0.1806, -0.0532, 0.2715, -0.1216,
					0.2236, 0.0406, 0.2140, 0.0467, -0.1651, 0.1224, 0.1095,
				],
			)
			.unwrap(),
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					-0.2333, -0.2742, -0.3514, 0.3495, 0.0522, 0.0418, 0.0179, -0.1399, 0.2596,
					0.2267, -0.2515, 0.1497, 0.2429, -0.0289, 0.0270, 0.0663,
				],
			)
			.unwrap(),
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					-0.2934, -0.2274, 0.3189, -0.3351, 0.0641, -0.3418, -0.1103, -0.0434, 0.0251,
					-0.1703, 0.1416, 0.0752, -0.2955, 0.0304, 0.3125, 0.2961,
				],
			)
			.unwrap(),
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					-0.1629, 0.2041, -0.2540, 0.3194, 0.1261, 0.1248, -0.2625, -0.1216, -0.3152,
					0.1714, -0.2976, -0.1349, -0.0290, 0.1442, 0.0928, 0.1064,
				],
			)
			.unwrap(),
		];

		let wv = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
		let wv_data = [
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					0.2772, 0.0084, -0.0406, -0.1352, -0.0876, -0.2885, 0.2212, 0.3212, 0.2180,
					0.1709, -0.0997, -0.2909, 0.3356, -0.1063, -0.1372, -0.2649,
				],
			)
			.unwrap(),
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					-0.2329, 0.0720, 0.2058, -0.3106, -0.3117, -0.0476, -0.2966, 0.2771, 0.0065,
					-0.2565, 0.0910, 0.2585, 0.0512, -0.2550, 0.0579, -0.2656,
				],
			)
			.unwrap(),
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					0.2022, -0.0342, 0.0798, -0.0472, -0.2836, 0.0980, 0.3008, -0.0706, -0.1285,
					-0.2878, -0.0419, 0.3432, -0.0951, -0.1046, 0.2416, -0.2881,
				],
			)
			.unwrap(),
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					-0.0520, -0.1166, -0.0103, -0.2266, 0.0090, 0.0456, 0.0630, -0.2110, 0.0817,
					-0.3090, 0.0070, 0.0388, -0.0943, -0.0283, 0.0018, 0.1874,
				],
			)
			.unwrap(),
		];
		let wo = g.alloc();
		let wo_data = Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size, hidden_size),
			vec![
				0.3145, -0.3200, 0.2741, 0.1205, 0.3450, 0.3156, -0.1222, -0.1644, 0.2381, 0.2044,
				0.2938, -0.1504, 0.0883, -0.1238, -0.2624, 0.2205, 0.0746, 0.2830, 0.1778, -0.2471,
				-0.1642, 0.1091, 0.2670, -0.2952, 0.3398, 0.2191, -0.0275, -0.3388, -0.1491,
				0.2444, 0.1212, 0.0551, -0.2984, -0.3337, -0.3091, 0.0010, -0.0480, -0.1975,
				0.3358, -0.3465, -0.3336, -0.3103, -0.1746, 0.0245, -0.1024, 0.0264, 0.2697,
				-0.2675, -0.1618, 0.3360, -0.0767, -0.1261, 0.2304, 0.3236, -0.1146, -0.2632,
				0.3489, 0.2969, 0.1339, 0.0510, -0.0413, -0.1972, -0.3190, 0.3461,
			],
		)
		.unwrap();

		let rsqrt = g.alloc();
		let rsqrt_data = Array4::<f32>::from_shape_vec(
			(1, 1, word_num, word_num),
			vec![
				1. / ((hidden_size / 4) as f32).sqrt(),
				1. / ((hidden_size / 4) as f32).sqrt(),
				1. / ((hidden_size / 4) as f32).sqrt(),
				1. / ((hidden_size / 4) as f32).sqrt(),
			],
		)
		.unwrap();

		let att = build_4_head_attention(&mut g, input, wq, wk, wv, rsqrt, wo);

		let (res, grad) = g.run(vec![
			(input, input_data.clone()),
			(wq[0], wq_data[0].clone()),
			(wq[1], wq_data[1].clone()),
			(wq[2], wq_data[2].clone()),
			(wq[3], wq_data[3].clone()),
			(wk[0], wk_data[0].clone()),
			(wk[1], wk_data[1].clone()),
			(wk[2], wk_data[2].clone()),
			(wk[3], wk_data[3].clone()),
			(wv[0], wv_data[0].clone()),
			(wv[1], wv_data[1].clone()),
			(wv[2], wv_data[2].clone()),
			(wv[3], wv_data[3].clone()),
			(wo, wo_data.clone()),
			(rsqrt, rsqrt_data.clone()),
		]);
		assert!(is_equal(
			res[att].iter(),
			[
				0.1555, 0.1530, 0.3986, 0.3954, 0.1307, 0.1349, -0.0585, -0.0530, -0.2580, -0.2566,
				-0.1810, -0.1798, -0.3973, -0.3936, 0.2313, 0.2283
			]
			.iter()
		));
		assert!(is_equal(
			grad[wq[0]].iter(),
			[
				0.0051, 0.0034, -0.0057, 0.0069, 0.0057, 0.0030, 0.0075, -0.0043, -0.0004, -0.0003,
				0.0005, -0.0006, -0.0005, -0.0003, -0.0006, 0.0004
			]
			.iter()
		));
		assert!(is_equal(
			grad[wq[1]].iter(),
			[
				0.0002, 0.0002, -0.0003, 0.0003, 0.0003, 0.0001, 0.0004, -0.0002, 0.0014, 0.0009,
				-0.0015, 0.0019, 0.0015, 0.0008, 0.0020, -0.0011
			]
			.iter()
		));
		assert!(is_equal(
			grad[wk[2]].iter(),
			[
				2.2342e-02,
				1.3491e-02,
				-1.5923e-02,
				1.8221e-02,
				-8.6214e-03,
				7.2143e-04,
				-1.9323e-02,
				-1.8840e-04,
				3.6030e-03,
				2.1756e-03,
				-2.5679e-03,
				2.9384e-03,
				-1.3903e-03,
				1.1634e-04,
				-3.1161e-03,
				-3.0384e-05
			]
			.iter()
		));
		assert!(is_equal(
			grad[wv[3]].iter(),
			[
				-0.2339, -0.1573, 0.2545, -0.3115, -0.2393, -0.1297, -0.3067, 0.1848, 0.9541,
				0.6415, -1.0379, 1.2704, 0.9758, 0.5291, 1.2507, -0.7535
			]
			.iter()
		));
	}

	#[test]
	#[ignore = "Can't pass pytorch implementation reference code as of now"]
	fn test_mha2() {
		/*REFERENCE CODE
		import torch
		import torch.nn as nn
		import torch.nn.functional as F

		torch.manual_seed(12)

		num_words=2
		hidden_size = 8
		num_heads = 4

		example_input = torch.rand((num_words, hidden_size))
		mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, bias=False, batch_first=True, )

		output = mha(example_input,example_input,example_input)

		print(len(mha.state_dict()['in_proj_weight']))

		print(attention_module)
		print("\nInput:")
		print(example_input)
		print("\nOutput:")
		print(output)

		print("\nWq:")
		print(mha.state_dict()['in_proj_weight'][0:8])
		print("\nWk:")
		print(mha.state_dict()['in_proj_weight'][8:16])
		print("\nWv:")
		print(mha.state_dict()['in_proj_weight'][16:24])

		print("\nWo:")
		print(mha.state_dict()['out_proj.weight'])*/

		let mut g = ComputationGraph::new();
		let word_num = 2;
		let hidden_size = 8;

		let input = g.alloc();
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

		let wq = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
		let wq_data = [
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
					-0.2857, -0.3358, -0.4303, 0.4280, 0.0639, 0.0512, 0.0219, -0.1714, 0.3180,
					0.2776, -0.3080, 0.1833, 0.2975, -0.0354, 0.0331, 0.0812,
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

		let wk = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
		let wk_data = [
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					0.3395, 0.0103, -0.0497, -0.1656, -0.1073, -0.3533, 0.2709, 0.3934, 0.2670,
					0.2093, -0.1221, -0.3563, 0.4111, -0.1301, -0.1681, -0.3245,
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

		let wv = [g.alloc(), g.alloc(), g.alloc(), g.alloc()];
		let wv_data = [
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					0.3851, -0.3919, 0.3357, 0.1476, 0.4226, 0.3866, -0.1497, -0.2014, 0.2916,
					0.2503, 0.3599, -0.1842, 0.1081, -0.1516, -0.3214, 0.2700,
				],
			)
			.unwrap(),
			Array4::<f32>::from_shape_vec(
				(1, 1, hidden_size / 4, hidden_size),
				vec![
					0.0914, 0.3467, 0.2178, -0.3026, -0.2011, 0.1336, 0.3270, -0.3615, 0.4162,
					0.2683, -0.0337, -0.4149, -0.1826, 0.2993, 0.1484, 0.0675,
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
		let wo = g.alloc();
		let wo_data = Array4::<f32>::from_shape_vec(
			(1, 1, hidden_size, hidden_size),
			vec![
				0.3387, 0.2434, -0.2648, -0.0385, 0.1132, -0.3144, -0.2423, 0.2218, 0.1567,
				-0.1614, -0.1412, 0.0777, 0.0554, 0.0766, -0.0468, 0.2696, -0.1261, -0.1694,
				-0.1721, -0.2212, 0.1007, -0.2273, -0.2521, 0.1761, 0.1608, -0.2375, -0.1221,
				-0.2659, 0.0805, -0.0329, 0.1880, -0.2263, -0.1175, 0.3200, 0.2771, 0.3436, 0.0953,
				0.2695, 0.3105, -0.2706, -0.2586, 0.3115, 0.1275, 0.0393, 0.2626, -0.2982, 0.2530,
				0.1796, 0.1200, 0.0578, -0.0828, 0.1529, 0.2779, 0.0422, -0.1554, -0.1785, -0.0185,
				-0.2612, -0.2105, 0.0981, -0.2829, 0.3263, 0.0247, 0.1514,
			],
		)
		.unwrap()
		.permuted_axes([0, 1, 3, 2]);

		let rsqrt = g.alloc();
		let rsqrt_data = Array4::<f32>::from_shape_vec(
			(1, 1, word_num, word_num),
			vec![
				1. / ((hidden_size / 4) as f32).sqrt(),
				1. / ((hidden_size / 4) as f32).sqrt(),
				1. / ((hidden_size / 4) as f32).sqrt(),
				1. / ((hidden_size / 4) as f32).sqrt(),
			],
		)
		.unwrap();

		let att = build_4_head_attention(&mut g, input, wq, wk, wv, rsqrt, wo);

		let (res, grad) = g.run(vec![
			(input, input_data.clone()),
			(wq[0], wq_data[0].clone()),
			(wq[1], wq_data[1].clone()),
			(wq[2], wq_data[2].clone()),
			(wq[3], wq_data[3].clone()),
			(wk[0], wk_data[0].clone()),
			(wk[1], wk_data[1].clone()),
			(wk[2], wk_data[2].clone()),
			(wk[3], wk_data[3].clone()),
			(wv[0], wv_data[0].clone()),
			(wv[1], wv_data[1].clone()),
			(wv[2], wv_data[2].clone()),
			(wv[3], wv_data[3].clone()),
			(wo, wo_data.clone()),
			(rsqrt, rsqrt_data.clone()),
		]);
		dbg!(&res[att]);
		assert!(is_equal(
			res[att].iter(),
			[
				0.2414, 0.0504, -0.1067, -0.1029, -0.0720, 0.0117, -0.1269, -0.0106, 0.2435,
				0.0500, -0.1044, -0.1021, -0.0761, 0.0098, -0.1276, -0.0107
			]
			.iter()
		));
		// assert!(is_equal(
		// 	grad[wq[0]].iter(),
		// 	[
		// 		0.0051, 0.0034, -0.0057, 0.0069, 0.0057, 0.0030, 0.0075, -0.0043, -0.0004, -0.0003,
		// 		0.0005, -0.0006, -0.0005, -0.0003, -0.0006, 0.0004
		// 	]
		// 	.iter()
		// ));
		// assert!(is_equal(
		// 	grad[wq[1]].iter(),
		// 	[
		// 		0.0002, 0.0002, -0.0003, 0.0003, 0.0003, 0.0001, 0.0004, -0.0002, 0.0014, 0.0009,
		// 		-0.0015, 0.0019, 0.0015, 0.0008, 0.0020, -0.0011
		// 	]
		// 	.iter()
		// ));
		// assert!(is_equal(
		// 	grad[wk[2]].iter(),
		// 	[
		// 		2.2342e-02,
		// 		1.3491e-02,
		// 		-1.5923e-02,
		// 		1.8221e-02,
		// 		-8.6214e-03,
		// 		7.2143e-04,
		// 		-1.9323e-02,
		// 		-1.8840e-04,
		// 		3.6030e-03,
		// 		2.1756e-03,
		// 		-2.5679e-03,
		// 		2.9384e-03,
		// 		-1.3903e-03,
		// 		1.1634e-04,
		// 		-3.1161e-03,
		// 		-3.0384e-05
		// 	]
		// 	.iter()
		// ));
		// assert!(is_equal(
		// 	grad[wv[3]].iter(),
		// 	[
		// 		-0.2339, -0.1573, 0.2545, -0.3115, -0.2393, -0.1297, -0.3067, 0.1848, 0.9541,
		// 		0.6415, -1.0379, 1.2704, 0.9758, 0.5291, 1.2507, -0.7535
		// 	]
		// 	.iter()
		// ));
	}
}
