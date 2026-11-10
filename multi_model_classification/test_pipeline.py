import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from multi_model_classification import config, merger, orchestrator, utils
from multi_model_classification.classifier import HuggingFaceClassifier, LLMClassifier
from multi_model_classification.config import ClassificationConfig


class FakeClassifier(LLMClassifier):
    seen = []

    def classify_single(self, prompt):
        self.seen.append(prompt)
        if self.model_name == "org/model2" and "Second" in prompt:
            return "other"
        return "typo_differences"


class PipelineTests(unittest.TestCase):
    def test_pipeline_resume_and_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = pd.DataFrame([
                {"pair_id": "one", "left_names": "First", "right_names": "First"},
                {"pair_id": "two", "left_names": "Second", "right_names": "Second"},
            ])
            data.to_parquet(root / "medium_pairs.parquet", index=False)
            results = root / "results"
            settings = ClassificationConfig(model_names=["org/model1", "org/model2"], device="cpu")
            FakeClassifier.seen = []
            with patch.object(config, "RESULTS_DIR", results), \
                 patch.object(merger, "RESULTS_DIR", results), \
                 patch.object(orchestrator, "DATASETS_DIR", root), \
                 patch.object(orchestrator, "HuggingFaceClassifier", FakeClassifier):
                orchestrator.run_full_pipeline(settings)
                agreed = pd.read_parquet(results / "consensus" / "agreed_pairs.parquet")
                review = pd.read_parquet(results / "consensus" / "disagreed_pairs.parquet")
                self.assertEqual(agreed["pair_id"].tolist(), ["one"])
                self.assertEqual(review["pair_id"].tolist(), ["two"])
                self.assertTrue(pd.isna(review["majority_type"].iloc[0]))
                self.assertTrue(all('"names":' in prompt for prompt in FakeClassifier.seen))
                self.assertEqual(len(FakeClassifier.seen), 4)
                orchestrator.run_full_pipeline(settings)
                self.assertEqual(len(FakeClassifier.seen), 4)
                data.loc[0, "left_names"] = "Changed"
                data.to_parquet(root / "medium_pairs.parquet", index=False)
                with self.assertRaisesRegex(ValueError, "Checkpoint prompt mismatch"):
                    orchestrator.run_full_pipeline(settings)

    def test_merge_matches_ids_and_writes_empty_parquet(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            with patch.object(config, "RESULTS_DIR", results), patch.object(merger, "RESULTS_DIR", results):
                for model, ids in [("a", ["1", "2"]), ("b", ["2", "1"])]:
                    path = utils.get_output_parquet_path(model)
                    pd.DataFrame({"pair_id": ids, "llm_pair_type": ["other", "other"]}).to_parquet(path)
                agreed, review = merger.merge_results(["a", "b"])
                merger.save_consensus_datasets(agreed, review, ["a", "b"])
                self.assertEqual(len(agreed), 2)
                self.assertEqual(len(pd.read_parquet(results / "consensus" / "disagreed_pairs.parquet")), 0)
                pd.DataFrame({"pair_id": ["1", "3"], "llm_pair_type": ["other", "other"]}).to_parquet(utils.get_output_parquet_path("b"))
                with self.assertRaisesRegex(ValueError, "Pair IDs do not match"):
                    merger.merge_results(["a", "b"])

    def test_invalid_responses_are_not_other(self):
        for reply in ("", "I cannot decide", "typo_differences or other"):
            with self.assertRaises(ValueError):
                utils.parse_label(reply)
        self.assertEqual(utils.parse_label("other"), "other")

    def test_huggingface_decodes_only_generated_tokens(self):
        class Inputs(dict):
            def to(self, device):
                return self

        class Tokenizer:
            chat_template = "present"
            pad_token = "pad"
            pad_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return "prompt"

            def __call__(self, text, **kwargs):
                return Inputs(input_ids=torch.tensor([[1, 2, 3]]))

            def decode(self, tokens, **kwargs):
                self.decoded = tokens.tolist()
                return "other"

        class Model:
            device = "cpu"

            def eval(self):
                return self

            def generate(self, **kwargs):
                return torch.tensor([[1, 2, 3, 4, 5]])

        tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(config, "RESULTS_DIR", Path(directory)), \
             patch("multi_model_classification.classifier.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("multi_model_classification.classifier.AutoModelForCausalLM.from_pretrained", return_value=Model()) as loader:
            classifier = HuggingFaceClassifier("test/model", ClassificationConfig(["test/model"], device="cpu"))
            self.assertEqual(classifier.classify_single("test prompt"), "other")
            self.assertEqual(tokenizer.decoded, [4, 5])
            self.assertNotIn("load_in_8bit", loader.call_args.kwargs)
            self.assertEqual(tokenizer.messages[0]["content"], "test prompt")

    def test_cuda_uses_quantization_config(self):
        from unittest.mock import MagicMock

        tokenizer = MagicMock(chat_template="present", pad_token="pad")
        model = MagicMock()
        quantization = object()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(config, "RESULTS_DIR", Path(directory)), \
             patch("multi_model_classification.classifier.torch.cuda.is_available", return_value=True), \
             patch("multi_model_classification.classifier.importlib.util.find_spec", return_value=True), \
             patch("multi_model_classification.classifier.BitsAndBytesConfig", return_value=quantization) as quantizer, \
             patch("multi_model_classification.classifier.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("multi_model_classification.classifier.AutoModelForCausalLM.from_pretrained", return_value=model) as loader:
            HuggingFaceClassifier("test/model", ClassificationConfig(["test/model"], device="cuda"))
            quantizer.assert_called_once_with(load_in_8bit=True)
            self.assertIs(loader.call_args.kwargs["quantization_config"], quantization)
            self.assertNotIn("load_in_8bit", loader.call_args.kwargs)

    def test_real_transformers_load_and_generate_on_cpu(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "tiny-model"
            model_path.mkdir()
            tokenizer = Tokenizer(models.WordLevel(
                {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "prompt": 3, "other": 4},
                unk_token="[UNK]",
            ))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            hf_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
                chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}",
            )
            hf_tokenizer.save_pretrained(model_path)
            model = Qwen2ForCausalLM(Qwen2Config(
                vocab_size=5, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
            ))
            model.save_pretrained(model_path)
            with patch.object(config, "RESULTS_DIR", root / "results"), \
                 patch("multi_model_classification.classifier.AutoTokenizer.from_pretrained", return_value=hf_tokenizer):
                classifier = HuggingFaceClassifier(str(model_path), ClassificationConfig(
                    [str(model_path)], device="cpu", max_tokens=2,
                ))
                self.assertIsInstance(classifier.classify_single("prompt prompt"), str)


if __name__ == "__main__":
    unittest.main()
