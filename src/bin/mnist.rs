use dlrs::{
	graph::computation_graph::ComputationGraph,
	operation::{
		eltwise_add, eltwise_add_back, fully_connected, fully_connected_back, relu, relu_back,
		softmax_cross_entropy, softmax_cross_entropy_back,
	},
};
use mnist::{Mnist, MnistBuilder};
use ndarray::{s, Array4};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
/*
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

class SimpleNN(nn.Module):
	def __init__(self):
		super(SimpleNN, self).__init__()
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(28 * 28, 128)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(128, 10)
		#nn.CrossEntropyLoss()가 softmax를 씌우기 때문에 softmax는 여기서 생략한다고 함.

	def forward(self, x):
		x = self.flatten(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		return x
net = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
correct = 0
total = 0

# Testing the network before learning
with torch.no_grad():
	for data in testloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
print('Accuracy on the test images before learning: %d %%' % (100 * correct / total))

# Training loop
for epoch in range(5):  # loop over the dataset multiple times
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:  # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training')

# Testing the network after learning
with torch.no_grad():
	for data in testloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
print('Accuracy on the test images after learning: %d %%' % (100 * correct / total))
*/

fn main() {
	let chunk_size = 50;
	let learning_rate = 0.01;
	let mut g = ComputationGraph::new();
	let mut rng = StdRng::seed_from_u64(1234);

	let input = g.alloc();
	// let input_data =
	// 	Array4::<f32>::from_shape_fn((1, 1, 28 * 28, 1), |_| rng.gen_range(-0.1..=0.1));

	let weight1 = g.alloc();
	let mut weight1_data =
		Array4::<f32>::from_shape_fn((1, 1, 128, 28 * 28), |_| rng.gen_range(-0.1..=0.1));

	let weight2 = g.alloc();
	let mut weight2_data =
		Array4::<f32>::from_shape_fn((1, 1, 10, 128), |_| rng.gen_range(-0.1..=0.1));

	let bias1 = g.alloc();
	let mut bias1_data =
		Array4::<f32>::from_shape_fn((1, 1, 128, 1), |_| rng.gen_range(-0.1..=0.1));

	let bias2 = g.alloc();
	let mut bias2_data = Array4::<f32>::from_shape_fn((1, 1, 10, 1), |_| rng.gen_range(-0.1..=0.1));

	let truth = g.alloc();
	// let truth_data =
	// 	Array4::<f32>::from_shape_fn((1, 1, 10, 1), |(_, _, i, _)| if i == 3 { 1. } else { 0. });

	let fc1 = g.alloc();
	g.adj[fc1].op = (fully_connected, fully_connected_back);
	g.connect(weight1, fc1);
	g.connect(input, fc1);

	let eltw_add1 = g.alloc();
	g.adj[eltw_add1].op = (eltwise_add, eltwise_add_back);
	g.connect(fc1, eltw_add1);
	g.connect(bias1, eltw_add1);

	let relu1 = g.alloc();
	g.adj[relu1].op = (relu, relu_back);
	g.connect(eltw_add1, relu1);

	let fc2 = g.alloc();
	g.adj[fc2].op = (fully_connected, fully_connected_back);
	g.connect(weight2, fc2);
	g.connect(relu1, fc2);

	let eltw_add2 = g.alloc();
	g.adj[eltw_add2].op = (eltwise_add, eltwise_add_back);
	g.connect(fc2, eltw_add2);
	g.connect(bias2, eltw_add2);

	let smce = g.alloc();
	g.adj[smce].op = (softmax_cross_entropy, softmax_cross_entropy_back);
	g.connect(eltw_add2, smce);
	g.connect(truth, smce);

	const TRAIN_DATA_SIZE: usize = 50000;
	const VALIDATION_DATA_SIZE: usize = 10000;
	const TEST_DATA_SIZE: usize = 10000;
	let Mnist {
		trn_img,
		trn_lbl,
		val_img,
		val_lbl,
		tst_img,
		tst_lbl,
	} = MnistBuilder::new()
		.download_and_extract()
		.label_format_one_hot()
		.training_set_length(TRAIN_DATA_SIZE as u32)
		.validation_set_length(VALIDATION_DATA_SIZE as u32)
		.test_set_length(TEST_DATA_SIZE as u32)
		.finalize();

	let train_data = Array4::from_shape_vec((TRAIN_DATA_SIZE, 1, 28 * 28, 1), trn_img)
		.unwrap()
		.map(|x| *x as f32 / 256.0);
	let train_labels: Array4<f32> = Array4::from_shape_vec((TRAIN_DATA_SIZE, 1, 10, 1), trn_lbl)
		.unwrap()
		.map(|x| *x as f32);

	let validation_data = Array4::from_shape_vec((VALIDATION_DATA_SIZE, 1, 28 * 28, 1), val_img)
		.unwrap()
		.map(|x| *x as f32 / 256.0);
	let validation_labels: Array4<f32> =
		Array4::from_shape_vec((VALIDATION_DATA_SIZE, 1, 10, 1), val_lbl)
			.unwrap()
			.map(|x| *x as f32);

	let test_data = Array4::from_shape_vec((TEST_DATA_SIZE, 1, 28 * 28, 1), tst_img)
		.unwrap()
		.map(|x| *x as f32 / 256.0);
	let test_labels: Array4<f32> = Array4::from_shape_vec((TEST_DATA_SIZE, 1, 10, 1), tst_lbl)
		.unwrap()
		.map(|x| *x as f32);

	{
		println!("##### Test #####");
		let input_data = test_data.clone();
		let truth_data = test_labels.clone();
		let (res, grad) = g.run(vec![
			(input, input_data.clone()),
			(weight1, weight1_data.clone()),
			(weight2, weight2_data.clone()),
			(bias1, bias1_data.clone()),
			(bias2, bias2_data.clone()),
			(truth, truth_data.clone()),
		]);
		println!("Cross Entropy Error: {}", res[smce].mean().unwrap());
	}

	println!("##### Learn #####");
	for epoch in 0..5 {
		println!("----- epoch {} -----", epoch);
		let mut chunks = Vec::new();
		for i in 0..TRAIN_DATA_SIZE / chunk_size {
			let input_data = train_data
				.slice(s![i * chunk_size..(i + 1) * chunk_size, .., .., ..])
				.to_owned();
			let truth_data = train_labels
				.slice(s![i * chunk_size..(i + 1) * chunk_size, .., .., ..])
				.to_owned();
			chunks.push((input_data, truth_data));
		}
		chunks.shuffle(&mut rng);
		for (input_data, truth_data) in chunks {
			let (res, grad) = g.run(vec![
				(input, input_data.clone()),
				(weight1, weight1_data.clone()),
				(weight2, weight2_data.clone()),
				(bias1, bias1_data.clone()),
				(bias2, bias2_data.clone()),
				(truth, truth_data.clone()),
			]);
			weight1_data -= &(grad[weight1].clone() * learning_rate);
			weight2_data -= &(grad[weight2].clone() * learning_rate);
			bias1_data -= &(grad[bias1].clone() * learning_rate);
			bias2_data -= &(grad[bias2].clone() * learning_rate);
		}
		{
			let input_data = validation_data.clone();
			let truth_data = validation_labels.clone();
			let (res, grad) = g.run(vec![
				(input, input_data.clone()),
				(weight1, weight1_data.clone()),
				(weight2, weight2_data.clone()),
				(bias1, bias1_data.clone()),
				(bias2, bias2_data.clone()),
				(truth, truth_data.clone()),
			]);
			println!(
				"Cross Entropy Error (validation): {}",
				res[smce].mean().unwrap()
			);
		}
	}
	{
		println!("##### Test #####");
		let input_data = test_data.clone();
		let truth_data = test_labels.clone();
		let (res, grad) = g.run(vec![
			(input, input_data.clone()),
			(weight1, weight1_data.clone()),
			(weight2, weight2_data.clone()),
			(bias1, bias1_data.clone()),
			(bias2, bias2_data.clone()),
			(truth, truth_data.clone()),
		]);
		println!("Cross Entropy Error (test): {}", res[smce].mean().unwrap());
	}
}
