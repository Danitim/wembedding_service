#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Word embeddings computation class."""

import json
import sys
import time
import urllib.request

import numpy as np


class WEmbeddings:
    """Class to keep multiple constructed word embedding computation models."""

    MODELS_MAP = {
        # Key: model name. Value: transformer model name, layer start, layer end.
        "bert-base-multilingual-uncased-last4": ("bert-base-multilingual-uncased", -4, None),
        "rubert-conversational-last4": ("DeepPavlov/rubert-base-cased-conversational", -4, None),
        "modernbert-base-last4": ("answerdotai/ModernBERT-base", -4, None),
        "robeczech-base-last4": ("ufal/robeczech-base", -4, None),
        "xlm-roberta-base-last4": ("xlm-roberta-base", -4, None),
    }

    MAX_SUBWORDS_PER_SENTENCE = 510

    class _Model:
        """Construct a tokenizer and transformers model graph."""
        def __init__(self, transformers_model, layer_start, layer_end, loader_lock):
            self._model_loaded = False
            self._transformers_model_name = transformers_model
            self._layer_start = layer_start
            self._layer_end = layer_end
            self._loader_lock = loader_lock

        def _load_tf(self):
            """Load model using TensorFlow backend."""
            import tensorflow as tf
            import transformers

            self._transformers_model = transformers.TFAutoModel.from_pretrained(
                self._transformers_model_name,
                config=transformers.AutoConfig.from_pretrained(self._transformers_model_name, output_hidden_states=True),
            )

            def compute_embeddings(subwords, segments):
                subword_embeddings_layers = self._transformers_model(
                    (tf.maximum(subwords, 0), tf.cast(tf.not_equal(subwords, -1), tf.int32))
                ).hidden_states
                subword_embeddings = tf.math.reduce_mean(subword_embeddings_layers[self._layer_start:self._layer_end], axis=0)

                # Average subwords (word pieces) word embeddings for each token
                def average_subwords(embeddings_and_segments):
                    subword_embeddings, segments = embeddings_and_segments
                    return tf.math.segment_mean(subword_embeddings, segments)
                word_embeddings = tf.map_fn(average_subwords, (subword_embeddings[:, 1:], segments), dtype=tf.float32)[:, :-1]
                return word_embeddings
            self.compute_embeddings = tf.function(compute_embeddings).get_concrete_function(
                tf.TensorSpec(shape=[None, None], dtype=tf.int32), tf.TensorSpec(shape=[None, None], dtype=tf.int32)
            )

        def _load_torch(self):
            """Load model using PyTorch backend."""
            import torch
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            import transformers

            self._torch_model = transformers.AutoModel.from_pretrained(
                self._transformers_model_name,
                config=transformers.AutoConfig.from_pretrained(self._transformers_model_name, output_hidden_states=True),
            )
            self._torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._torch_model.to(self._torch_device).eval()

            layer_start = self._layer_start
            layer_end = self._layer_end
            device = self._torch_device
            model = self._torch_model

            def compute_embeddings(np_subwords, np_segments):
                subwords = torch.from_numpy(np_subwords).to(device)
                input_ids = torch.clamp(subwords, min=0)
                attention_mask = (subwords != -1).long()

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # Average selected hidden layers
                hidden_states = outputs.hidden_states
                stacked = torch.stack(hidden_states[layer_start:layer_end])
                subword_embeddings = stacked.mean(dim=0)

                # Remove [CLS] token
                subword_embeddings = subword_embeddings[:, 1:, :]

                segments = torch.from_numpy(np_segments).to(device)
                batch_size, seq_len, hidden_size = subword_embeddings.shape
                max_words = int(segments.max().item()) + 1

                # segment_mean: average subword embeddings per word
                seg_expanded = segments.unsqueeze(-1).expand(-1, -1, hidden_size)
                word_sums = torch.zeros(batch_size, max_words, hidden_size, device=device)
                word_sums.scatter_add_(1, seg_expanded, subword_embeddings)

                counts = torch.zeros(batch_size, max_words, 1, device=device)
                counts.scatter_add_(1, segments.unsqueeze(-1), torch.ones(batch_size, seq_len, 1, device=device))
                counts = counts.clamp(min=1)

                word_embeddings = word_sums / counts
                # Remove the last segment (padding / [SEP])
                word_embeddings = word_embeddings[:, :-1, :]

                return word_embeddings.cpu()

            self.compute_embeddings = compute_embeddings

        def _add_special_tokens(self, token_ids):
            """Add [CLS] and [SEP] to a list of token ids.

            build_inputs_with_special_tokens is a no-op for some fast
            tokenizers (e.g. ModernBERT) that rely on the post_processor
            in tokenizer.json instead.  This helper always works.
            """
            result = self.tokenizer.build_inputs_with_special_tokens(token_ids)
            # If the method didn't actually add anything, do it manually.
            if len(result) == len(token_ids):
                cls_id = self.tokenizer.cls_token_id
                sep_id = self.tokenizer.sep_token_id
                if cls_id is not None and sep_id is not None:
                    result = [cls_id] + token_ids + [sep_id]
            return result

        def load(self):
            if self._model_loaded: return
            with self._loader_lock:
                import transformers

                if self._model_loaded: return

                self.tokenizer = transformers.AutoTokenizer.from_pretrained(self._transformers_model_name, use_fast=True)

                # Try TensorFlow first, fall back to PyTorch.
                try:
                    self._load_tf()
                except (ImportError, ValueError):
                    self._load_torch()

                self._model_loaded = True


    def __init__(self, max_form_len=64, threads=None, preload_models=[]):
        import threading

        # Impose the limit on the number of threads, if given (TF only)
        if threads is not None:
            try:
                import tensorflow as tf
                tf.config.threading.set_inter_op_parallelism_threads(threads)
                tf.config.threading.set_intra_op_parallelism_threads(threads)
            except ImportError:
                pass

        self._max_form_len = max_form_len

        loader_lock = threading.Lock()
        self._models = {}
        for model_name, (transformers_model, layer_start, layer_end) in self.MODELS_MAP.items():
            self._models[model_name] = self._Model(transformers_model, layer_start, layer_end, loader_lock)

            if model_name in preload_models or "all" in preload_models:
                self._models[model_name].load()

    def compute_embeddings(self, model, sentences):
        """Computes word embeddings.
        Arguments:
            model: one of the keys of self.MODELS_MAP.
            sentences: 2D Python array with sentences with tokens (strings).
        Returns:
            embeddings as a Python list of 1D Numpy arrays
        """

        if model not in self._models:
            print("No such WEmbeddings model {}".format(model), file=sys.stderr, flush=True)

        embeddings = []
        if sentences:
            model = self._models[model]
            model.load()

            time_tokenization = time.time()

            sentences_subwords = model.tokenizer(
                [(" " if i else "") + word[:self._max_form_len] for sentence in sentences for i, word in enumerate(sentence)],
                add_special_tokens=False
            ).input_ids

            subwords, segments, parts = [], [], []
            for sentence in sentences:
                segments.append([])
                subwords.append([])
                parts.append([0])
                sentence_subwords, sentences_subwords = sentences_subwords[:len(sentence)], sentences_subwords[len(sentence):]
                for word_subwords in sentence_subwords:
                    # Split sentences with too many subwords
                    if len(subwords[-1]) + len(word_subwords) > self.MAX_SUBWORDS_PER_SENTENCE:
                        subwords[-1] = model._add_special_tokens(subwords[-1])
                        segments.append([])
                        subwords.append([])
                        parts[-1].append(0)
                    segments[-1].extend([parts[-1][-1]] * len(word_subwords))
                    subwords[-1].extend(word_subwords)
                    parts[-1][-1] += 1
                subwords[-1] = model._add_special_tokens(subwords[-1])

            max_sentence_len = max(len(sentence) for sentence in sentences)
            max_subwords = max(len(sentence) for sentence in subwords)

            time_embeddings = time.time()
            np_subwords = np.full([len(subwords), max_subwords], -1, np.int32)
            for i, subword in enumerate(subwords):
                np_subwords[i, :len(subword)] = subword

            np_segments = np.full([len(segments), max_subwords - 1], max_sentence_len, np.int32)
            for i, segment in enumerate(segments):
                np_segments[i, :len(segment)] = segment

            embeddings_with_parts = model.compute_embeddings(np_subwords, np_segments).numpy()

            # Concatenate splitted sentences
            current_sentence_part = 0
            for sentence_parts in parts:
                embeddings.append(np.concatenate(
                    [embeddings_with_parts[current_sentence_part + i, :sentence_part] for i, sentence_part in enumerate(sentence_parts)],
                    axis=0))
                current_sentence_part += len(sentence_parts)

            print("WEmbeddings in {:.1f}ms,".format(1000 * (time.time() - time_embeddings)),
                  "tokenization in {:.1f}ms,".format(1000*(time_embeddings - time_tokenization)),
                  "batch {},".format(len(sentences)),
                  "max sentence len {},".format(max_sentence_len),
                  "max subwords {}.".format(max_subwords),
                  file=sys.stderr, flush=True)

        return embeddings


    class ClientNetwork:
        def __init__(self, url):
            self._url = url
        def compute_embeddings(self, model, sentences):
            with urllib.request.urlopen(
                    "http://{}/wembeddings".format(self._url),
                    data=json.dumps({"model": model, "sentences": sentences}, ensure_ascii=True).encode("ascii"),
            ) as response:
                embeddings = []
                for _ in sentences:
                    embeddings.append(np.lib.format.read_array(response, allow_pickle=False))
                return embeddings
