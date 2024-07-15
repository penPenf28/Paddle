# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from op_test import OpTest

# for func
import paddle
from paddle import _C_ops
from paddle.base.layer_helper import LayerHelper
from paddle.framework import (
    in_dynamic_mode,
)


def cache_evit(
    x, choice_index, head_num, topk, sink_tokens, proxy_tokens, random_keeps
):
    if in_dynamic_mode():
        print("dynamic")
        final_out = _C_ops.cache_evit(
            x,
            choice_index,
            head_num,
            topk,
            sink_tokens,
            proxy_tokens,
            random_keeps,
        )
        return final_out
    else:
        print("static")
        helper = LayerHelper('cache_evit', **locals())
        dtype = x.dtype
        # check dtypes
        # check_variable_and_dtype(
        #     x, 'x', ['float16'], 'cache_evit'
        # )
        # check_dtype(
        #     dtype,
        #     'dtype',
        #     ['float16'],
        #     'cache_evit',
        # )

        # set inputs
        inputs = {
            'x': x,
            'choice_index': choice_index,
        }

        # set attrs
        attrs = {
            'head_num': head_num,
            'topk': topk,
            'sink_tokens': sink_tokens,
            'proxy_tokens': proxy_tokens,
            'random_keeps': random_keeps,
        }

        final_out = helper.create_variable_for_type_inference(dtype=dtype)
        outputs = {
            'out': final_out,
        }

        helper.append_op(
            type='cache_evit',
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
        )

        return final_out


class TestCacheEvitOp(OpTest):
    def setUp(self):
        self.batch_size = 1
        self.num_head = 32
        self.seq_k = 2048
        self.topk = 512
        self.sink_tokens = 256
        self.proxy_tokens = 256
        # random_keeps = 512
        # 随机保留比例
        self.proxy_token_keep_ratio = 0.5
        self.random_keeps = (int)(
            self.proxy_token_keep_ratio
            * (self.seq_k - self.topk - self.sink_tokens - self.proxy_tokens)
        )

        self.__class__.op_type = "cache_evit"
        # use autograd to check grad in this unittest.
        self.__class__.no_need_check_grad = False

        self.x_type = np.float32
        paddle.set_default_dtype(self.x_type)

        self.GenerateInput()

    def GenerateInput(self):
        self.input_act = np.random.uniform(
            -10, 10, (self.batch_size, self.num_head, self.seq_k)
        ).astype(self.x_type)

        # 在768-1792之间取 512个随机值保留
        # left = self.sink_tokens + self.topk
        # right = self.seq_k - self.proxy_tokens
        # self.choice_index = np.random.randint(left, right + 1, size=self.random_keeps).tolist()

    def GetBaselineOut(self):
        tensor_input = paddle.to_tensor(self.input_act, stop_gradient=False)
        sink_tokens = self.sink_tokens
        proxy_tokens = self.proxy_tokens
        random_keeps = self.random_keeps

        sink_keep_idx = np.arange(sink_tokens)
        recent_keep_idx = np.arange(self.seq_k - proxy_tokens, self.seq_k)
        choice_index = []
        index = []

        for head_idx in range(self.num_head):
            proxy_score_cur_head = tensor_input[:, head_idx].squeeze()

            proxy_score_cur_head = proxy_score_cur_head[
                sink_tokens:-proxy_tokens
            ]
            topk_score, topk_idx = paddle.topk(
                proxy_score_cur_head, k=self.topk
            )
            proxy_keep_idx = topk_idx.numpy()

            to_evit_tokens_num = proxy_score_cur_head.shape[-1]
            idx_item_proxy_removed = np.delete(
                list(range(to_evit_tokens_num)), proxy_keep_idx
            )
            random_keep_idx = np.random.choice(
                idx_item_proxy_removed, size=self.random_keeps, replace=False
            )
            choice_index.append(random_keep_idx)

            proxy_keep_idx = proxy_keep_idx + sink_tokens
            random_keep_idx = random_keep_idx + sink_tokens

            index_item = np.concatenate(
                [
                    sink_keep_idx,
                    np.sort(
                        np.concatenate(
                            [proxy_keep_idx, random_keep_idx], axis=-1
                        )
                    ),
                    recent_keep_idx,
                ],
                axis=-1,
            )
            index.append(index_item)

        index = paddle.to_tensor(index).reshape(
            [self.batch_size, self.num_head, -1]
        )

        self.choice_index = choice_index

        output = paddle.take_along_axis(tensor_input, index, axis=-1)
        return output

    def GetFusedCacheEvit(self):
        # paddle.enable_static()
        # 输入为归一化后的结果
        tensor_input = paddle.to_tensor(self.input_act, stop_gradient=False)
        topk = self.topk
        num_head = self.num_head
        # choice_index size为[bs*num_head*random_keeps]
        choice_index = paddle.to_tensor(
            self.choice_index, stop_gradient=False
        ).astype(paddle.int32)
        print(f'choice_index.shape:{choice_index.shape}')
        sink_tokens = self.sink_tokens
        proxy_tokens = self.proxy_tokens
        random_keeps = self.random_keeps

        output = cache_evit(
            tensor_input,
            choice_index,
            num_head,
            topk,
            sink_tokens,
            proxy_tokens,
            random_keeps,
        )
        return output

    # def test_baseline(self):
    #     ans = self.GetBaselineOut()
    #     print(self.input_act)
    #     print(ans)

    def test_cache_evit_op(self):
        ref_output = self.GetBaselineOut()
        print(ref_output[0, 5, 100])
        op_output = self.GetFusedCacheEvit()
        print(op_output[0, 5, 100])
        np.testing.assert_allclose(ref_output, op_output, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
