import argparse
import pandas as pd
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

list_of_methods = ['character','keyboard','random','substitute','swap','delete','spell']

augs = {'character': nac.OcrAug(), 'keyboard': nac.KeyboardAug(), 'random': nac.RandomCharAug(), \
		'substitute': nac.RandomCharAug(action='substitute'), 'swap': nac.RandomCharAug(action='swap'), 'delete': nac.RandomCharAug(action='delete'),\
		'spell': naw.SpellingAug()}

if __name__ == '__main__':
	parser = argparse.ArgumentParser(prog="Inject Artifical Noise to Text Data", conflict_handler='resolve')
	parser.add_argument('--source_path', type=str, help='Path to the source file', required=True)
	parser.add_argument('--target_path', type=str, help='Path to the target file', required=False, default=None)
	parser.add_argument('--output_file', type=str, help='Path to the output file', required=True)
	parser.add_argument('--injection_method', type=str, help='Noise Injection Method', required=True)

	args, _ = parser.parse_known_args()

	if args.injection_method not in list_of_methods:
		raise ValueError("Injection Method not in list {}".format(list_of_methods))
	else:
		method = augs[args.injection_method]

	texts = open(args.source_path,'r').readlines()
	texts = [text.replace('\n','') for text in texts]

	augmented_texts = [method.augment(text, n=1) for text in texts]

	df = pd.DataFrame()
	df['source'] = augmented_texts

	if args.target_path:
		targets = open(args.target_path,'r').readlines()
		targets = [text.replace('\n','') for text in targets]
		df['target'] = targets
	else:
		df['target'] = texts

	df.to_csv(args.output_file, index=False, sep='\t')



