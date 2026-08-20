def test_import():
    import spec4ml_py
    import spec4ml_py.preprocessing

    assert hasattr(spec4ml_py, "__version__")
    assert spec4ml_py.__version__ == "0.2.0"
