import json
import os
from konlpy.tag import Kkma

class Paragraph:
	def __init__(self, paragraph_txt):
		self.paragraph_txt = paragraph_txt

class Score:
	def __init__(self, essay_scoreT_avg):
		self.essay_scoreT_avg = essay_scoreT_avg

class Info:
	def __init__(self, essay_prompt, essay_main_subject):
		self.essay_prompt = essay_prompt
		self.essay_main_subject = essay_main_subject

class EssayData:
	def __init__(self, paragraph, score, info):
		self.paragraph = [Paragraph(**p) for p in paragraph]
		self.score = Score(**score)
		self.info = Info(**info)

input_directory_path = "./data/essay/train_test"
output_directory_path = "./data/essay/trainp"
parser = Kkma()
for filename in os.listdir(input_directory_path):
	input_file_path = os.path.join(input_directory_path, filename)
	output_file_path = os.path.join(output_directory_path, filename)
	if os.path.isfile(input_file_path):
		with open(input_file_path, 'r') as file:
			s = file.read()
			essay_data_dict = json.loads(s)
			essay_data = EssayData(**essay_data_dict)
			
			text = "".join(i.paragraph_txt for i in essay_data.paragraph).replace("#@문장구분#", "#")
			res_para = "".join(i[0]+"@" for i in parser.pos(text))

			text = essay_data.info.essay_prompt.replace("#@문장구분#", "#")
			res_prompt = "".join(i[0]+"@" for i in parser.pos(text))

		with open(output_file_path, 'w', encoding='utf-8') as json_file:
			json.dump({
				"prompt": res_prompt,
				"paragraph": res_para,
				"score": essay_data.score.essay_scoreT_avg
			}, json_file, ensure_ascii=False)