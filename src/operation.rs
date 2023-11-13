use ndarray::{Array4, Axis};

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
//input[0]=input
//output[0]=input_gradient_local
pub fn identity_back(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
	vec![input[0].clone()]
}

//input[0]=input
//input[1]=weight
//output[0]=output
pub fn fully_connected(input: &Vec<Array4<f32>>) -> Array4<f32> {
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
	assert!(b0 == b1 && f0 == f1 && x0 == y1);
	let mut ret = Array4::zeros((b0, f0, y0, x1));
	// let a = array![[3, 4], [1, 2]].row_;
	for bi in 0..b0 {
		for fi in 0..f0 {
			ret.index_axis_mut(Axis(0), bi)
				.index_axis_mut(Axis(0), fi)
				.assign(
					&input[0]
						.index_axis(Axis(0), bi)
						.index_axis(Axis(0), fi)
						.dot(&input[1].index_axis(Axis(0), bi).index_axis(Axis(0), fi)),
				);
		}
	}
	ret
}
//input[0]=input
//input[1]=weight
//output[0]=input_grad
//output[0]=weight_grad_local
pub fn fully_connected_back(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
	todo!();
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

//input[0]=input
//output[0]=input_grad_local
pub fn relu_back(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
	let (b, f, y, x) = (
		input[0].shape()[0],
		input[0].shape()[1],
		input[0].shape()[2],
		input[0].shape()[3],
	);
	let mask = input[0].mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
	vec![mask; 1]
}
