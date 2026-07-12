from agentconnect.common.compression import Compressor


def test_short_text_passes_through_unchanged():
    c = Compressor(min_chars_to_compress=500)
    text = "a short string"
    out, stats = c.compress_text(text, "prose")
    assert out == text
    assert stats.original_chars == stats.compressed_chars
    assert stats.ratio == 0.0


def test_kind_not_in_apply_to_passes_through():
    c = Compressor(min_chars_to_compress=1, apply_to=("prose",))
    text = "x" * 1000
    out, _stats = c.compress_text(text, "tool_output")
    assert out == text


def test_code_fence_preserved_byte_for_byte():
    c = Compressor(min_chars_to_compress=10)
    code = "```python\ndef f():\n    pass\n```"
    text = "It is worth noting that this needs review.\n\n" + code + "\nI would recommend testing it."
    out, _stats = c.compress_text(text, "prose")
    assert code in out
    assert "It is worth noting that" not in out
    assert "I would recommend" not in out


def test_url_preserved_exactly_once():
    c = Compressor(min_chars_to_compress=10)
    text = "It is worth noting that " + ("padding " * 20) + "see https://example.com/a/b?x=1 for details."
    out, _stats = c.compress_text(text, "prose")
    assert out.count("https://example.com/a/b?x=1") == 1


def test_repeated_lines_collapsed_above_threshold():
    c = Compressor(min_chars_to_compress=10)
    lines = ["OBSERVATION:"] + (["waiting..."] * 10) + ["done"]
    text = "\n".join(lines)
    out, stats = c.compress_text(text, "tool_output")
    assert "[7 repeated lines omitted]" in out
    assert stats.compressed_chars < stats.original_chars


def test_repeated_lines_below_threshold_untouched():
    c = Compressor(min_chars_to_compress=10)
    lines = ["OBSERVATION:", "waiting...", "waiting...", "done"]  # run of 2 < min_run of 4
    text = "\n".join(lines)
    out, _stats = c.compress_text(text, "tool_output")
    assert out == text


def test_progress_lines_stripped():
    c = Compressor(min_chars_to_compress=10)
    text = "\n".join(["OBSERVATION:", "Build starting", "[####      ] 40%", "...45%", "Build complete"])
    out, _stats = c.compress_text(text, "tool_output")
    assert "40%" not in out
    assert "45%" not in out
    assert "Build complete" in out


def test_ansi_codes_stripped():
    c = Compressor(min_chars_to_compress=10)
    text = "OBSERVATION:\n" + "\x1b[32mOK\x1b[0m" + "\npadding" * 10
    out, _stats = c.compress_text(text, "tool_output")
    assert "\x1b[" not in out
    assert "OK" in out


def test_per_provider_disable():
    c = Compressor(min_chars_to_compress=5, per_provider={"p1": {"enabled": False}})
    text = "It is worth noting that " + "x" * 100
    out, _stats = c.compress_for_provider("p1", text, "prose")
    assert out == text


def test_stats_accumulate_per_provider():
    c = Compressor(min_chars_to_compress=5)
    c.compress_for_provider("p1", "It is worth noting that " + "x" * 100, "prose")
    c.compress_for_provider("p1", "It is worth noting that " + "y" * 100, "prose")
    stats = c.stats_for("p1")
    assert stats["original_chars"] > 0
    assert stats["compressed_chars"] < stats["original_chars"]
    assert stats["ratio"] > 0
    assert c.stats_for("p2") == {"original_chars": 0, "compressed_chars": 0, "ratio": 0.0}


def test_from_config_defaults():
    c = Compressor.from_config(None)
    assert c.enabled is True
    assert c.apply_to == ("tool_output", "prose")

    c2 = Compressor.from_config({"enabled": False, "min_chars_to_compress": 42})
    assert c2.enabled is False
    assert c2.min_chars_to_compress == 42


def test_inflation_guard_returns_original(monkeypatch):
    c = Compressor(min_chars_to_compress=1, apply_to=("prose",))
    original = "compact input"
    monkeypatch.setattr(
        "agentconnect.common.compression._compress_prose",
        lambda text: text + " expanded",
    )

    out, stats = c.compress_text(original, "prose")

    assert out == original
    assert stats.original_chars == stats.compressed_chars == len(original)
