import numpy as np
from pgvector import HalfVector
import pytest
from struct import pack


class TestHalfVector:
    def test_list(self):
        assert HalfVector([1, 2, 3]).to_list() == [1, 2, 3]

    def test_list_str(self):
        with pytest.raises(ValueError, match='could not convert string to float'):
            HalfVector([1, 'two', 3])

    def test_tuple(self):
        assert HalfVector((1, 2, 3)).to_list() == [1, 2, 3]

    def test_ndarray(self):
        arr = np.array([1, 2, 3])
        assert HalfVector(arr).to_list() == [1, 2, 3]
        assert HalfVector(arr).to_numpy() is not arr

    def test_ndarray_same_object(self):
        arr = np.array([1, 2, 3], dtype='>f2')
        assert HalfVector(arr).to_list() == [1, 2, 3]
        assert HalfVector(arr).to_numpy() is arr

    def test_ndim_two(self):
        with pytest.raises(ValueError) as error:
            HalfVector([[1, 2], [3, 4]])
        assert str(error.value) == 'expected ndim to be 1'

    def test_ndim_zero(self):
        with pytest.raises(ValueError) as error:
            HalfVector(1)
        assert str(error.value) == 'expected ndim to be 1'

    def test_repr(self):
        assert repr(HalfVector([1, 2, 3])) == 'HalfVector([1.0, 2.0, 3.0])'
        assert str(HalfVector([1, 2, 3])) == 'HalfVector([1.0, 2.0, 3.0])'

    def test_equality(self):
        assert HalfVector([1, 2, 3]) == HalfVector([1, 2, 3])
        assert HalfVector([1, 2, 3]) != HalfVector([1, 2, 4])

    def test_dimensions(self):
        assert HalfVector([1, 2, 3]).dimensions() == 3

    def test_from_text(self):
        vec = HalfVector.from_text('[1.5,2,3]')
        assert vec.to_list() == [1.5, 2, 3]
        assert np.array_equal(vec.to_numpy(), [1.5, 2, 3])

    def test_from_binary(self):
        data = pack('>HH3e', 3, 0, 1.5, 2, 3)
        vec = HalfVector.from_binary(data)
        assert vec.to_list() == [1.5, 2, 3]
        assert np.array_equal(vec.to_numpy(), [1.5, 2, 3])
        assert vec.to_binary() == data

    def test_to_text(self):
        vec = HalfVector([1.5, 2, 3])
        assert vec.to_text() == '[1.5,2.0,3.0]'

    def test_to_db(self):
        vec = HalfVector([1, 2, 3])
        assert HalfVector._to_db(vec) == '[1.0,2.0,3.0]'

    def test_to_db_list(self):
        assert HalfVector._to_db([1, 2, 3]) == '[1.0,2.0,3.0]'

    def test_to_db_none(self):
        assert HalfVector._to_db(None) is None

    def test_to_db_dim(self):
        vec = HalfVector([1, 2, 3])
        assert HalfVector._to_db(vec, 3) == '[1.0,2.0,3.0]'

    def test_to_db_dim_invalid(self):
        vec = HalfVector([1, 2, 3])
        with pytest.raises(ValueError, match='expected 2 dimensions, not 3'):
            HalfVector._to_db(vec, 2)

    def test_to_db_binary(self):
        vec = HalfVector([1, 2, 3])
        result = HalfVector._to_db_binary(vec)
        assert result == vec.to_binary()

    def test_to_db_binary_list(self):
        result = HalfVector._to_db_binary([1, 2, 3])
        assert result == HalfVector([1, 2, 3]).to_binary()

    def test_to_db_binary_none(self):
        assert HalfVector._to_db_binary(None) is None

    def test_from_db_text(self):
        result = HalfVector._from_db('[1.5,2,3]')
        assert result.to_list() == [1.5, 2, 3]

    def test_from_db_none(self):
        assert HalfVector._from_db(None) is None

    def test_from_db_halfvector(self):
        vec = HalfVector([1, 2, 3])
        assert HalfVector._from_db(vec) is vec

    def test_from_db_binary(self):
        data = pack('>HH3e', 3, 0, 1.5, 2, 3)
        result = HalfVector._from_db_binary(data)
        assert result.to_list() == [1.5, 2, 3]

    def test_from_db_binary_none(self):
        assert HalfVector._from_db_binary(None) is None

    def test_from_db_binary_halfvector(self):
        vec = HalfVector([1, 2, 3])
        assert HalfVector._from_db_binary(vec) is vec
