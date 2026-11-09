from minisgl_mlx.message import DetokenizeMsg
from minisgl_mlx.tokenizer import DetokenizeManager
from minisgl_mlx.utils import load_tokenizer


def test_incremental_detokenize_matches_full_decode(model_path):
    tok = load_tokenizer(model_path)
    eos = tok.eos_token_id
    manager = DetokenizeManager(tok, {eos})
    texts = ["你好，世界！今天天气怎么样？🌤️ Let's go.", "def f(x):\n    return x ** 2  # 平方"]
    for uid, text in enumerate(texts):
        ids = tok.encode(text)
        streamed = ""
        for i, t in enumerate(ids + [eos]):
            streamed += manager.detokenize([DetokenizeMsg(uid, t, finished=(i == len(ids)))])[0]
        assert streamed == tok.decode(ids)
        assert "�" not in streamed
    assert manager.decode_map == {}
