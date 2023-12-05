#[cfg(test)]
mod tests {
	use ndarray::Array4;

	use crate::{
		computation_graph::ComputationGraph,
		graph_builder::build_attention,
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
	fn test_encoder() {
		/*REFERENCE CODE
		TODO*/
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
}