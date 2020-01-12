from synth.generation import TextGenerator


class TextcatGenerationAugmenter:
    def __init__(
        self,
        tg: TextGenerator,
        train_data: List,
        dev_data: List,
        n_gen: int = 2,
        max_length: int = 50,
        aug_ratio: float = 0.2
    ):
        self.tg = tg
        self.train_data = train_data
        self.dev_data = dev_data
        self.n_gen = n_gen
        self.max_length = max_length
        self.aug_ratio = aug_ratio

    def augment(self):
        train = self.train_data
        random.shuffle(train)
        augment_base = train[len(train) * self.aug_ratio]

        for e in tqdm(augment_base):
            text = e['text']
            cats = e['']
            cats = e[1]
            
            doc = nlp(text)
            if len(doc) > max_length:
                prompt_text = list(doc.sents)[0].text
            else:
                prompt_text = doc.text

            for i in range(n_gen):
                gen_text = tg.run_generation(
                    prompt_text,
                    length=randint(20, 50),
                    repetition_penalty=i + 1.0,
                    stop_token='<|endoftext|>'
                )
                generated_examples.append((gen_text, cats))
                srsly.write_jsonl('./generated_examples.jsonl', generated_examples)
                self.train_data
