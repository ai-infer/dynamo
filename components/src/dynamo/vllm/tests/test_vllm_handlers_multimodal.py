# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from PIL import Image

import pytest

from dynamo.vllm.handlers import _compute_mm_uuids

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def test_compute_mm_uuids_supports_kimi_vision_chunk():
    image = Image.new("RGB", (2, 2), color="blue")

    mm_uuids = _compute_mm_uuids(
        {
            "vision_chunk": [
                {
                    "type": "image",
                    "image": image,
                    "uuid": None,
                }
            ]
        }
    )

    assert mm_uuids is not None
    assert "vision_chunk" in mm_uuids
    assert len(mm_uuids["vision_chunk"]) == 1
