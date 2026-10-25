from ipomdp.infra.resources import process_rss_mb


def test_process_rss_is_positive_and_grows_with_allocation():
    before = process_rss_mb()
    block = bytearray(64 * 2**20)
    block[::4096] = b"x" * len(block[::4096])
    assert before > 0
    assert process_rss_mb() > before + 32
