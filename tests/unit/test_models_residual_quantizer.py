import torch

from openthought2text.models import ResidualVectorQuantizer, ResidualVectorQuantizerConfig


def test_residual_quantizer_multiple_levels_shapes_encode_decode_and_gradients():
    torch.manual_seed(10)
    quantizer = ResidualVectorQuantizer(
        ResidualVectorQuantizerConfig(num_codebooks=3, codebook_size=5, embedding_dim=4)
    )
    embeddings = torch.randn(2, 4, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    output = quantizer(embeddings, mask)
    assert output.indices.shape == (2, 4, 3)
    assert output.quantized.shape == embeddings.shape
    assert len(output.per_level_health) == 3
    assert all(item.usage.shape == (5,) for item in output.per_level_health)
    assert torch.all(output.quantized[~mask] == 0)
    torch.testing.assert_close(quantizer.decode(output.indices, mask), output.quantized.detach())
    torch.testing.assert_close(quantizer.encode(embeddings, mask), output.indices)
    (output.quantized.square().mean() + output.loss).backward()
    assert embeddings.grad is not None and embeddings.grad.abs().sum() > 0
    assert all(codebook.weight.grad is not None for codebook in quantizer.codebooks)


def test_residual_quantizer_reports_per_level_codebook_collapse():
    quantizer = ResidualVectorQuantizer(
        ResidualVectorQuantizerConfig(num_codebooks=2, codebook_size=4, embedding_dim=2)
    )
    with torch.no_grad():
        for codebook in quantizer.codebooks:
            codebook.weight.zero_()
    output = quantizer(torch.ones(1, 3, 2))
    assert output.indices.unique().tolist() == [0]
    for health in output.per_level_health:
        assert health.perplexity.item() == 1.0
        assert health.active_fraction.item() == 0.25
        assert health.dead_fraction.item() == 0.75


def test_residual_decode_rejects_wrong_level_count_and_out_of_range_codes():
    quantizer = ResidualVectorQuantizer(
        ResidualVectorQuantizerConfig(num_codebooks=2, codebook_size=3, embedding_dim=2)
    )
    try:
        quantizer.decode(torch.zeros(1, 2, 1, dtype=torch.long))
    except ValueError as error:
        assert "num_codebooks" in str(error)
    else:
        raise AssertionError("expected invalid level count to fail")
    try:
        quantizer.decode(torch.full((1, 2, 2), 3, dtype=torch.long))
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("expected invalid code to fail")
