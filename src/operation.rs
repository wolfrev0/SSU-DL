use ndarray::{array, Array4, Axis, Shape};

//input[0]=input
//output[0]=output
pub fn identity(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
	input.clone()
}

//input[0]=input
//input[1]=weight
//output[0]=output
pub fn fully_connected(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
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
	vec![ret; 1]
}
//input[0]=input
//input[1]=weight
//input[2]=error
//output[0]=error_out
pub fn fully_connected_back(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
	todo!();
}

//input[0]=input
//output[0]=output
pub fn relu(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
	assert!(input.len() == 1);
	let mut ret = input[0].clone();
	for i in ret.iter_mut() {
		*i = i.max(0.)
	}
	vec![ret; 1]
}

//input[0]=input
//input[1]=error
//output[0]=error_out
pub fn relu_back(input: &Vec<Array4<f32>>) -> Vec<Array4<f32>> {
	assert!(input.len() == 1);
	let mut ret = input[0].clone();
	for i in ret.iter_mut() {
		*i = i.max(0.)
	}
	vec![ret; 1]
}
