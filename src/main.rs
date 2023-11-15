use dlrs::{
	graph::computation_graph::ComputationGraph,
	misc::util::is_equal,
	operation::{
		eltwise_add, eltwise_add_back, fully_connected, fully_connected_back, relu, relu_back,
	},
};
use ndarray::Array4;
use rand::Rng;
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
	let mut rng = rand::thread_rng();

	let mut g = ComputationGraph::new();

	let input = g.alloc();
	let input_data =
		Array4::<f32>::from_shape_fn((1, 1, 28 * 28, 1), |_| rng.gen_range(-0.1..=0.1));

	let weight1 = g.alloc();
	let weight1_data =
		Array4::<f32>::from_shape_fn((1, 1, 128, 28 * 28), |_| rng.gen_range(-0.1..=0.1));

	let weight2 = g.alloc();
	let weight2_data = Array4::<f32>::from_shape_fn((1, 1, 10, 128), |_| rng.gen_range(-0.1..=0.1));

	let bias1 = g.alloc();
	let bias1_data = Array4::<f32>::from_shape_fn((1, 1, 128, 1), |_| rng.gen_range(-0.1..=0.1));

	let bias2 = g.alloc();
	let bias2_data = Array4::<f32>::from_shape_fn((1, 1, 10, 1), |_| rng.gen_range(-0.1..=0.1));

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

	let (res, grad) = g.run(vec![
		(input, input_data.clone()),
		(weight1, weight1_data.clone()),
		(weight2, weight2_data.clone()),
		(bias1, bias1_data.clone()),
		(bias2, bias2_data.clone()),
	]);
	dbg!(&res[eltw_add2]);
}
