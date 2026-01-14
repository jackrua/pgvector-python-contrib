import numpy as np
from pgvector import Vector
import pytest
from struct import pack


class TestVector:
    def test_list(self):
        assert Vector([1, 2, 3]).to_list() == [1, 2, 3]

    def test_list_str(self):
        with pytest.raises(ValueError, match='could not convert string to float'):
            Vector([1, 'two', 3])

    def test_tuple(self):
        assert Vector((1, 2, 3)).to_list() == [1, 2, 3]

    def test_ndarray(self):
        arr = np.array([1, 2, 3])
        assert Vector(arr).to_list() == [1, 2, 3]
        assert Vector(arr).to_numpy() is not arr

    def test_ndarray_same_object(self):
        arr = np.array([1, 2, 3], dtype='>f4')
        assert Vector(arr).to_list() == [1, 2, 3]
        assert Vector(arr).to_numpy() is arr

    def test_ndim_two(self):
        with pytest.raises(ValueError) as error:
            Vector([[1, 2], [3, 4]])
        assert str(error.value) == 'expected ndim to be 1'

    def test_ndim_zero(self):
        with pytest.raises(ValueError) as error:
            Vector(1)
        assert str(error.value) == 'expected ndim to be 1'

    def test_repr(self):
        assert repr(Vector([1, 2, 3])) == 'Vector([1.0, 2.0, 3.0])'
        assert str(Vector([1, 2, 3])) == 'Vector([1.0, 2.0, 3.0])'

    def test_equality(self):
        assert Vector([1, 2, 3]) == Vector([1, 2, 3])
        assert Vector([1, 2, 3]) != Vector([1, 2, 4])

    def test_dimensions(self):
        assert Vector([1, 2, 3]).dimensions() == 3

    def test_from_text(self):
        vec = Vector.from_text('[1.5,2,3]')
        assert vec.to_list() == [1.5, 2, 3]
        assert np.array_equal(vec.to_numpy(), [1.5, 2, 3])

    def test_from_binary(self):
        data = pack('>HH3f', 3, 0, 1.5, 2, 3)
        vec = Vector.from_binary(data)
        assert vec.to_list() == [1.5, 2, 3]
        assert np.array_equal(vec.to_numpy(), [1.5, 2, 3])
        assert vec.to_binary() == data

    def test_to_text(self):
        vec = Vector([1, 2, 3])
        assert vec.to_text() == '[1.0,2.0,3.0]'

    def test_to_db_none(self):
        assert Vector._to_db(None) is None

    def test_to_db_vector(self):
        vec = Vector([1, 2, 3])
        assert Vector._to_db(vec) == '[1.0,2.0,3.0]'

    def test_to_db_list(self):
        assert Vector._to_db([1, 2, 3]) == '[1.0,2.0,3.0]'

    def test_to_db_with_dim(self):
        assert Vector._to_db([1, 2, 3], 3) == '[1.0,2.0,3.0]'

    def test_to_db_wrong_dim(self):
        with pytest.raises(ValueError, match='expected 3 dimensions, not 2'):
            Vector._to_db([1, 2], 3)

    def test_to_db_binary_none(self):
        assert Vector._to_db_binary(None) is None

    def test_to_db_binary_vector(self):
        vec = Vector([1, 2, 3])
        result = Vector._to_db_binary(vec)
        assert result == pack('>HH3f', 3, 0, 1, 2, 3)

    def test_to_db_binary_list(self):
        result = Vector._to_db_binary([1, 2, 3])
        assert result == pack('>HH3f', 3, 0, 1, 2, 3)

    def test_from_db_none(self):
        assert Vector._from_db(None) is None

    def test_from_db_ndarray(self):
        arr = np.array([1, 2, 3], dtype=np.float32)
        assert Vector._from_db(arr) is arr

    def test_from_db_text(self):
        result = Vector._from_db('[1.5,2,3]')
        expected = np.array([1.5, 2, 3], dtype=np.float32)
        assert np.array_equal(result, expected)

    def test_from_db_binary_none(self):
        assert Vector._from_db_binary(None) is None

    def test_from_db_binary_ndarray(self):
        arr = np.array([1, 2, 3], dtype=np.float32)
        assert Vector._from_db_binary(arr) is arr

    def test_from_db_binary_bytes(self):
        data = pack('>HH3f', 3, 0, 1.5, 2, 3)
        result = Vector._from_db_binary(data)
        expected = np.array([1.5, 2, 3], dtype=np.float32)
        assert np.array_equal(result, expected)

    def test_empty_vector(self):
        vec = Vector([])
        assert vec.dimensions() == 0
        assert vec.to_list() == []

    def test_single_element(self):
        vec = Vector([42])
        assert vec.dimensions() == 1
        assert vec.to_list() == [42]

    def test_negative_values(self):
        vec = Vector([-1, -2, -3])
        assert vec.to_list() == [-1, -2, -3]

    def test_float_precision(self):
        vec = Vector([1.123456789, 2.987654321])
        # Float32 precision test
        result = vec.to_list()
        assert len(result) == 2
        assert abs(result[0] - 1.123456789) < 1e-6
        assert abs(result[1] - 2.987654321) < 1e-6
