from snippy import core


def test_source_code():
    def greet():
        return "Hello, World!"

    actual = core.source_code(greet)
    expected = '    def greet():\n        return "Hello, World!"\n'
    assert actual == expected
