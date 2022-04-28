# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import set_rng_seed
from torch import nn
from torchmultimodal.modules.layers.quantisation import Quantisation


class TestQuantisation(unittest.TestCase):
    """
    Test the Quantisation class
    """

    def setUp(self):
        torch.set_printoptions(precision=10)
        set_rng_seed(4)
        self.num_embeddings = 4
        self.embedding_dim = 5
        # This is 2x5x3
        self.encoded = torch.Tensor(
            [
                [[-1, 0, 1], [2, 1, 0], [0, -1, -1], [0, 2, -1], [-2, -1, 1]],
                [[2, 2, -1], [1, -1, -2], [0, 0, 0], [1, 2, 1], [1, 0, 0]],
            ]
        )
        # This is 4x5
        self.embedding_weights = torch.Tensor(
            [[1, 0, -1, -1, 2], [2, -2, 0, 0, 1], [2, 1, 0, 1, 1], [-1, -2, 0, 2, 0]]
        )
        # This is 4x3
        self.test_tensor_flat = torch.Tensor(
            [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        )

    def test_quantised_output(self):
        vq = Quantisation(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        )
        vq.embedding = nn.Embedding.from_pretrained(self.embedding_weights)
        actual = vq(self.encoded)

        # This is shape (2,5,3)
        expected = torch.Tensor(
            [
                [
                    [2.0, 2.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 1.0, -1.0],
                    [1.0, 1.0, 2.0],
                ],
                [
                    [2.0, 2.0, -1.0],
                    [1.0, -2.0, -2.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 2.0],
                    [1.0, 1.0, 0.0],
                ],
            ]
        )

        torch.testing.assert_close(
            actual,
            expected,
            msg=f"actual: {actual}, expected: {expected}",
        )

    def test_quantised_shape(self):
        vq = Quantisation(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        )
        vq.embedding = nn.Embedding.from_pretrained(self.embedding_weights)
        output = vq(self.encoded)
        actual = torch.tensor(output.shape)
        expected = torch.tensor([2, 5, 3])

        assert torch.equal(
            actual, expected
        ), f"actual shape: {actual}, expected shape: {expected}"

    def test_preprocess(self):
        vq = Quantisation(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        )
        encoded_flat, permuted_shape = vq._preprocess(self.encoded)

        expected_flat_shape = torch.tensor([6, 5])
        expected_permuted_shape = torch.tensor([2, 3, 5])

        actual_flat_shape = torch.tensor(encoded_flat.shape)
        actual_permuted_shape = torch.tensor(permuted_shape)

        assert torch.equal(
            actual_flat_shape, expected_flat_shape
        ), f"actual flattened shape: {actual_flat_shape}, expected flattened shape: {expected_flat_shape}"

        assert torch.equal(
            actual_permuted_shape, expected_permuted_shape
        ), f"actual permuted shape: {actual_permuted_shape}, expected permuted shape: {expected_permuted_shape}"

    def test_preprocess_channel_dim_assertion(self):
        vq = Quantisation(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        )
        with self.assertRaises(Exception):
            encoded_flat, permuted_shape = vq._preprocess(self.encoded[:, :4, :])

    def test_postprocess(self):
        vq = Quantisation(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        )
        quantised = vq._postprocess(self.test_tensor_flat, torch.Size([2, 2, 3]))
        actual_quantised_shape = torch.tensor(quantised.shape)
        expected_quantised_shape = torch.tensor([2, 3, 2])

        assert torch.equal(
            actual_quantised_shape, expected_quantised_shape
        ), f"actual quantised shape: {actual_quantised_shape}, expected quantised shape: {expected_quantised_shape}"
