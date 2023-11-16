use ndarray::{s, Array3, Array4, Axis};

//fwd와 bwd 설명
//fwd(input_1, input_2, ..., input_N) -> output
//bwd(input_1, input_2, ..., input_N, sum_output_gradients) -> (grad_1, grad_2, ..., grad_N)
//fwd와 bwd를 하나로 합치는게 성능상 더 낫나? 근데 그러면 grad는 local기준이라서 어짜피 dfs한번 더 해줘야해서 비슷한가?
//fwd의 연산량이 많다면 합치고 local gradient사용한 dfs 한번 더 해주는게 성능상 좋을듯 한데 일단 넘어가자

//input[0]=input
//output[0]=output
pub fn identity(input: &Vec<Array4<f32>>) -> Array4<f32> {
	input[0].clone()
}
//input[-1]=output_grad_sum
//input[0]=input
//output[0]=input_gradient
pub fn identity_back(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
	vec![input[0].clone()]
}

fn bfyx_matmul(a: &Array4<f32>, b: &Array4<f32>) -> Array4<f32> {
	// let mut ret = Array4::zeros((b0, f0, y0, x1));
	// for bi in 0..b0 {
	// 	for fi in 0..f0 {
	// 		ret.index_axis_mut(Axis(0), bi)
	// 			.index_axis_mut(Axis(0), fi)
	// 			.assign(
	// 				&input[0]
	// 					.index_axis(Axis(0), bi)
	// 					.index_axis(Axis(0), fi)
	// 					.dot(&input[1].index_axis(Axis(0), bi).index_axis(Axis(0), fi)),
	// 			);
	// 	}
	// }
	let (b0, f0, y0, x0) = (a.shape()[0], a.shape()[1], a.shape()[2], a.shape()[3]);
	let (b1, f1, y1, x1) = (b.shape()[0], b.shape()[1], b.shape()[2], b.shape()[3]);
	assert!(b0 == b1 && f0 == f1 && x0 == y1);
	let mut ret = Array4::zeros((a.shape()[0], a.shape()[1], a.shape()[2], b.shape()[3]));
	for i in 0..b0 {
		for j in 0..f0 {
			let cur = a.slice(s![i, j, .., ..]).dot(&b.slice(s![i, j, .., ..]));
			// Update the result array
			ret.slice_mut(s![i, j, .., ..]).assign(&cur);
		}
	}
	ret
}

//input[0]=input
//input[1]=weight
//output[0]=output
pub fn fully_connected(input: &Vec<Array4<f32>>) -> Array4<f32> {
	bfyx_matmul(&input[0], &input[1])
}

//input[-1]=output_grad_sum
//input[0]=input
//input[1]=weight
//output[0]=input_grad
//output[1]=weight_grad
pub fn fully_connected_back(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
	vec![
		bfyx_matmul(&input[2], &input[1].clone().permuted_axes([0, 1, 3, 2])),
		bfyx_matmul(&input[0].clone().permuted_axes([0, 1, 3, 2]), &input[2]),
	]
}

//input[0]=input
//output[0]=output
pub fn relu(input: &Vec<Array4<f32>>) -> Array4<f32> {
	assert!(input.len() == 1);
	let mut ret = input[0].clone();
	for i in ret.iter_mut() {
		*i = i.max(0.)
	}
	ret
}

//input[-1]=output_grad_sum
//input[0]=input
//output[0]=input_grad
pub fn relu_back(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
	let mask = input[0].mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
	vec![mask * &input[1]; 1]
}

//input[0]=input0
//input[1]=input1
//output[0]=output
pub fn eltwise_add(input: &Vec<Array4<f32>>) -> Array4<f32> {
	input[0].clone() + &input[1]
}

//input[-1]=output_grad_sum
//input[0]=input0
//input[1]=input1
//output[0]=input0_grad
//output[0]=input1_grad
pub fn eltwise_add_back(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
	vec![input[2].clone(), input[2].clone()]
}

//input[0]=input
//output[0]=output
pub fn softmax(input: &Vec<Array4<f32>>) -> Array4<f32> {
	//TODO: support b>1 f>1 case
	let ret = input[0].map(|x| x.exp());
	let denom = ret.sum();
	ret / denom
}

pub fn softmax_back(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
	todo!();
}

//input[0]=input
//input[1]=truth
//output[0]=output
pub fn cross_entropy(input: &Vec<Array4<f32>>) -> Array4<f32> {
	//TODO: support b>1 f>1 case
	Array4::from_elem((1, 1, 1, 1), (input[0].map(|x| -x.ln()) * &input[1]).sum())
}

//input[-1]=output_grad_sum
//input[0]=input
//input[1]=truth
//output[0]=input_grad
//output[0]=truth_grad
pub fn cross_entropy_back(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
	//TODO: support b>1 f>1 case
	vec![
		-input[1].clone() / &input[0] * &input[2],
		input[0].map(|x| -x.ln()) * &input[2],
	]
}

//input[0]=input
//input[1]=truth
//output[0]=output
pub fn softmax_cross_entropy(input: &Vec<Array4<f32>>) -> Array4<f32> {
	let (b0, f0, y0, x0) = (
		input[0].shape()[0],
		input[0].shape()[1],
		input[0].shape()[2],
		input[0].shape()[3],
	);
	let (b1, f1, y1, x1) = (
		input[1].shape()[0],
		input[1].shape()[1],
		input[1].shape()[2],
		input[1].shape()[3],
	);
	assert!(b0 == b1 && f0 == f1 && y0 == y1 && x0 == x1);
	let mut ret = Array4::zeros((b0, 1, 1, 1));
	for i in 0..b0 {
		let m0exp = input[0].slice(s![i, .., .., ..]).map(|x| x.exp());
		let m1 = input[1].slice(s![i, .., .., ..]);
		let denom = m0exp.sum();
		let softmax_out = m0exp / denom;
		ret.slice_mut(s![i, .., .., ..]).assign(&Array3::from_elem(
			(1, 1, 1),
			(softmax_out.map(|x| -x.ln()) * &m1).sum(),
		));
	}
	ret
}

//input[-1]=output_grad_sum
//input[0]=input
//input[1]=truth
//output[0]=input_grad
//output[0]=truth_grad
pub fn softmax_cross_entropy_back(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
	let (b0, f0, y0, x0) = (
		input[0].shape()[0],
		input[0].shape()[1],
		input[0].shape()[2],
		input[0].shape()[3],
	);
	let (b1, f1, y1, x1) = (
		input[1].shape()[0],
		input[1].shape()[1],
		input[1].shape()[2],
		input[1].shape()[3],
	);
	let (b2, f2, y2, x2) = (1, 1, 1, 1);
	assert!(b0 == b1 && f0 == f1 && y0 == y1 && x0 == x1);
	let mut ret0 = Array4::zeros((b0, 1, y1, x2));
	let mut ret1 = Array4::zeros((b0, 1, y0, x2));
	for i in 0..b0 {
		let m0 = input[0].slice(s![i, .., .., ..]);
		let m1 = input[1].slice(s![i, .., .., ..]);
		let m2 = input[2].slice(s![i, .., .., ..]);
		let denom = m0.map(|x| x.exp()).sum();
		let softmax_out = m0.map(|x| x.exp()) / denom;
		ret0.slice_mut(s![i, .., .., ..])
			.assign(&((softmax_out - &m1) * &m2));
		ret1.slice_mut(s![i, .., .., ..])
			.assign(&(m0.map(|x| -x.ln()) * &m2));
	}
	vec![ret0, ret1]
}
