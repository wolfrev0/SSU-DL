use std::{
	fs::{self, File},
	io::{Read, Write},
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
	let directory_path = "./data/essay/train";
	let dir_entries = fs::read_dir(directory_path).unwrap();
	for entry in dir_entries {
		let entry = entry.unwrap();
		let file_path = entry.path();
		if file_path.is_file() {
			let mut file = File::open(&file_path).unwrap();
			let mut s = String::new();
			file.read_to_string(&mut s).unwrap();

			let essay_data: EssayData = serde_json::from_str(&s).unwrap();
			// println!("{:#?}", essay_data);

			let output_file_path = format!(
				"./data/essay/train_test/{}.json",
				file_path.file_stem().unwrap().to_string_lossy()
			);
			let output_file = File::create(&output_file_path).unwrap();
			serde_json::to_writer(output_file, &essay_data).unwrap();
			println!("Saved to: {}", output_file_path);
		}
	}
}
