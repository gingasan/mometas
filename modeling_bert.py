import torch
from torch import nn
from transformers import BertForMaskedLM
from transformers.models.bert.modeling_bert import BertModel, BertOnlyMLMHead, BertPooler


class BertForMultiObjectiveLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.atd = nn.Linear(config.hidden_size, 2)
        self.dtd = nn.Linear(config.hidden_size, 2)
        self.cse = BertPooler(config)

        self.init_weights()

    def get_parameters(self):
        return self.bert.parameters()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        flow="mlm"
    ):
        if flow == "cse":
            input_ids = input_ids.view((-1, input_ids.size(-1)))
            if attention_mask is not None:
                attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output = outputs[0]

        # Mask Language Modeling
        if flow == "mlm" or flow == "emlm":
            prediction_scores = self.cls(sequence_output)

            mlm_loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            return {
                "loss": mlm_loss,
                "logits": prediction_scores,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions
            }
        # Added Token Detection
        elif flow == "atd":
            prediction_scores = self.atd(sequence_output)

            atd_loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                atd_loss = loss_fct(prediction_scores.view(-1, 2), labels.view(-1))

            return {
                "loss": atd_loss,
                "logits": prediction_scores,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions
            }
        # Deleted Token Detection
        elif flow == "dtd":
            prediction_scores = self.dtd(sequence_output)

            dtd_loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                dtd_loss = loss_fct(prediction_scores.view(-1, 2), labels.view(-1))

            return {
                "loss": dtd_loss,
                "logits": prediction_scores,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions
            }
        # Contrastive learning of Sentence Embeddings
        elif flow == "cse":
            prediction_scores = self.cse(sequence_output)

            sim_fct = nn.CosineSimilarity(dim=-1)
            sim_scores = sim_fct(prediction_scores.unsqueeze(1), prediction_scores.unsqueeze(0))
            sim_scores = sim_scores - torch.eye(prediction_scores.shape[0], device=prediction_scores.device) * 1e12
            sim_scores = sim_scores / 0.05

            labels = torch.arange(prediction_scores.shape[0], device=prediction_scores.device)
            labels = (labels - labels % 2 * 2) + 1
            loss_fct = nn.CrossEntropyLoss()
            cse_loss = loss_fct(sim_scores, labels)

            return {
                "loss": cse_loss,
                "logits": prediction_scores,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions
            }

        else:
            raise NotImplementedError
