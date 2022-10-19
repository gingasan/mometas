import random
import torch


class Processor(object):
    def __init__(self, tokenizer, msl):
        self.tokenizer = tokenizer
        self.msl = msl
        self.sos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, inputs, task, **kwargs):
        if task == "mlm":
            return self.mask_tokens(inputs, **kwargs)
        elif task == "emlm":
            return self.mask_tokens_guided(inputs, **kwargs)
        elif task == "atd":
            return self.add_tokens(inputs, **kwargs)
        elif task == "dtd":
            return self.delete_tokens(inputs, **kwargs)
        elif task == "cse":
            return inputs[:, None, :].repeat(1, 2, 1), None
        else:
            raise NotImplementedError

    def mask_tokens(self, inputs, noise_probability=0.15):
        inputs, labels = inputs.clone(), inputs.clone()
        probability_matrix = torch.full(labels.shape, noise_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def mask_tokens_guided(self, inputs, guide_mask, noise_probability=0.5):
        inputs, labels = inputs.clone(), inputs.clone()
        probability_matrix = torch.zeros(labels.shape)

        probability_matrix.masked_fill_(guide_mask, value=float(noise_probability))
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def add_tokens(self, inputs, noise_probability=0.15):
        probability_matrix = torch.full(inputs.shape, noise_probability)

        noisy_inputs = []
        labels = []
        masked_indices = torch.bernoulli(probability_matrix).int().tolist()
        for i in range(len(masked_indices)):
            noisy_inputs += [[]]
            labels += [[]]
            j = 0
            k = 0
            while j < len(masked_indices[0]):
                if inputs[i][k] == 0:
                    noisy_inputs[-1] += [self.pad_id] * (self.msl - j)
                    labels[-1] += [0] * (self.msl - j)
                    break

                if j == 0 or j == len(masked_indices[0]) - 1:
                    labels[-1] += [0]
                    noisy_inputs[-1] += [inputs[i][k]]
                else:
                    label = masked_indices[i][j]
                    labels[-1] += [label]
                    if label == 0:
                        noisy_inputs[-1] += [inputs[i][k]]
                    else:
                        random_word = random.randint(1, len(self.tokenizer) - 1)
                        noisy_inputs[-1] += [random_word]
                        k -= 1

                j += 1
                k += 1

            while len(noisy_inputs[-1]) > self.msl:
                del noisy_inputs[-1][-1]
                del labels[-1][-1]
            if noisy_inputs[-1][-1] != self.pad_id:
                noisy_inputs[-1][-1] = self.eos_id

        noisy_inputs = torch.tensor(noisy_inputs, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return noisy_inputs, labels

    def delete_tokens(self, inputs, noise_probability=0.15):
        probability_matrix = torch.full(inputs.shape, noise_probability)

        noisy_inputs = []
        labels = []
        masked_indices = torch.bernoulli(probability_matrix).int().tolist()
        for i in range(len(masked_indices)):
            noisy_inputs += [[]]
            labels += [[]]
            j = 0
            while j < len(masked_indices[0]):
                if inputs[i][j] == 0:
                    noisy_inputs[-1] += [self.pad_id] * (self.msl - j)
                    labels[-1] += [0] * (self.msl - j)
                    break

                if j == 0 or j == len(masked_indices[0]) - 1:
                    labels[-1] += [0]
                    noisy_inputs[-1] += [inputs[i][j]]
                else:
                    label = masked_indices[i][j]
                    if len(labels[-1]) <= len(noisy_inputs[-1]):
                        labels[-1] += [label]
                    if label == 0:
                        noisy_inputs[-1] += [inputs[i][j]]

                j += 1

            while len(noisy_inputs[-1]) < self.msl:
                noisy_inputs[-1] += [self.pad_id]
            while len(labels[-1]) < self.msl:
                labels[-1] += [0]

        noisy_inputs = torch.tensor(noisy_inputs, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return noisy_inputs, labels
