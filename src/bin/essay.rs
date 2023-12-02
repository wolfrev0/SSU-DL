use std::{
	fs::{self, File},
	io::Read,
};

extern crate serde;
extern crate serde_json;

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct Paragraph {
	paragraph_txt: String,
}
#[derive(Debug, Deserialize, Serialize)]
struct Score {
	essay_scoreT_avg: f64,
}
#[derive(Debug, Deserialize, Serialize)]
struct Info {
	essay_prompt: String,
	essay_main_subject: String,
}
#[derive(Debug, Deserialize, Serialize)]
struct EssayData {
	paragraph: Vec<Paragraph>,
	score: Score,
	info: Info,
}

fn main() {
	println!("Reading data");
	//ko3.vec이 200만 라인
	//100만라인: OK(ko4)
	//130만라인: FAIL
	let mut file = File::open("data/ko4.vec").unwrap();

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

	let directory_path = "./data/essay/train";
	let dir_entries = fs::read_dir(directory_path).unwrap();
	for entry in dir_entries.take(10) {
		let entry = entry.unwrap();
		let file_path = entry.path();
		if file_path.is_file() {
			let mut file = File::open(&file_path).unwrap();
			let mut s = String::new();
			file.read_to_string(&mut s).unwrap();

			let essay_data: EssayData = serde_json::from_str(&s).unwrap();
			println!("{:#?}", essay_data);
		}
	}
}
