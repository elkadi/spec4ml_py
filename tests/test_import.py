def test_import():
    import spec4ml_py
    assert hasattr(spec4ml_py, '__version__')
