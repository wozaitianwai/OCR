import importlib
import os
import sys
import types
import numpy as np

def load_base_utils():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    cv2_stub = types.ModuleType('cv2')
    cv2_stub.createLineSegmentDetector = lambda *a, **k: None
    cv2_stub.getStructuringElement = lambda *a, **k: None
    cv2_stub.Canny = lambda *a, **k: None
    cv2_stub.dilate = lambda *a, **k: None
    cv2_stub.erode = lambda *a, **k: None
    cv2_stub.adaptiveThreshold = lambda *a, **k: np.zeros((1,1), dtype=np.uint8)
    cv2_stub.medianBlur = lambda img, size: img
    cv2_stub.bitwise_and = lambda a, b: None
    cv2_stub.add = lambda a, b: None
    cv2_stub.boundingRect = lambda cnt: [0,0,1,1]
    cv2_stub.boxPoints = lambda rect: np.array([[0,0],[1,0],[1,1],[0,1]])
    cv2_stub.cvtColor = lambda img, code: img
    cv2_stub.MORPH_RECT = cv2_stub.MORPH_OPEN = None
    cv2_stub.morphologyEx = lambda *a, **k: None
    sys.modules['cv2'] = cv2_stub

    pymysql_stub = types.ModuleType('pymysql')
    pymysql_stub.install_as_MySQLdb = lambda: None
    sys.modules['pymysql'] = pymysql_stub

    sqlalchemy_stub = types.ModuleType('sqlalchemy')
    sqlalchemy_stub.create_engine = lambda *a, **k: None
    sqlalchemy_stub.text = lambda x: x
    sys.modules['sqlalchemy'] = sqlalchemy_stub

    sqlalchemy_orm_stub = types.ModuleType('sqlalchemy.orm')
    sqlalchemy_orm_stub.sessionmaker = lambda bind=None: lambda: None
    sys.modules['sqlalchemy.orm'] = sqlalchemy_orm_stub

    psutil_stub = types.ModuleType('psutil')
    psutil_stub.Process = lambda pid: types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=0, vms=0))
    sys.modules['psutil'] = psutil_stub

    if 'tools.base_utils' in sys.modules:
        del sys.modules['tools.base_utils']
    return importlib.import_module('tools.base_utils')

def test_drop_duplicated_points():
    base_utils = load_base_utils()
    data = [4,5,6,7,8,9,10,1748,1749,1750,1751,1762,1763,1764,1765,1766,3504,3505,3506,3507,3508,3509]
    expected = [10, 1751, 1766, 3509]
    assert base_utils.drop_duplicated_points(data, max_span=10) == expected
