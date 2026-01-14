import numpy as np
from pgvector import SparseVector
import pytest
from scipy.sparse import coo_array, coo_matrix, csr_array, csr_matrix
from struct import pack


class TestSparseVector:
    def test_list(self):
        vec = SparseVector([1, 0, 2, 0, 3, 0])
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert np.array_equal(vec.to_numpy(), [1, 0, 2, 0, 3, 0])
        assert vec.indices() == [0, 2, 4]

    def test_list_dimensions(self):
        with pytest.raises(ValueError) as error:
            SparseVector([1, 0, 2, 0, 3, 0], 6)
        assert str(error.value) == 'extra argument'

    def test_ndarray(self):
        vec = SparseVector(np.array([1, 0, 2, 0, 3, 0]))
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_dict(self):
        vec = SparseVector({2: 2, 4: 3, 0: 1, 3: 0}, 6)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_dict_no_dimensions(self):
        with pytest.raises(ValueError) as error:
            SparseVector({0: 1, 2: 2, 4: 3})
        assert str(error.value) == 'missing dimensions'

    def test_coo_array(self):
        arr = coo_array(np.array([1, 0, 2, 0, 3, 0]))
        vec = SparseVector(arr)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_coo_array_dimensions(self):
        with pytest.raises(ValueError) as error:
            SparseVector(coo_array(np.array([1, 0, 2, 0, 3, 0])), 6)
        assert str(error.value) == 'extra argument'

    def test_coo_matrix(self):
        mat = coo_matrix(np.array([1, 0, 2, 0, 3, 0]))
        vec = SparseVector(mat)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_dok_array(self):
        arr = coo_array(np.array([1, 0, 2, 0, 3, 0])).todok()
        vec = SparseVector(arr)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_csr_array(self):
        arr = csr_array(np.array([[1, 0, 2, 0, 3, 0]]))
        vec = SparseVector(arr)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_csr_matrix(self):
        mat = csr_matrix(np.array([1, 0, 2, 0, 3, 0]))
        vec = SparseVector(mat)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_repr(self):
        assert repr(SparseVector([1, 0, 2, 0, 3, 0])) == 'SparseVector({0: 1.0, 2: 2.0, 4: 3.0}, 6)'
        assert str(SparseVector([1, 0, 2, 0, 3, 0])) == 'SparseVector({0: 1.0, 2: 2.0, 4: 3.0}, 6)'

    def test_equality(self):
        assert SparseVector([1, 0, 2, 0, 3, 0]) == SparseVector([1, 0, 2, 0, 3, 0])
        assert SparseVector([1, 0, 2, 0, 3, 0]) != SparseVector([1, 0, 2, 0, 3, 1])
        assert SparseVector([1, 0, 2, 0, 3, 0]) == SparseVector({2: 2, 4: 3, 0: 1, 3: 0}, 6)
        assert SparseVector({}, 1) != SparseVector({}, 2)

    def test_dimensions(self):
        assert SparseVector([1, 0, 2, 0, 3, 0]).dimensions() == 6

    def test_indices(self):
        assert SparseVector([1, 0, 2, 0, 3, 0]).indices() == [0, 2, 4]

    def test_values(self):
        assert SparseVector([1, 0, 2, 0, 3, 0]).values() == [1, 2, 3]

    def test_to_coo(self):
        assert np.array_equal(SparseVector([1, 0, 2, 0, 3, 0]).to_coo().toarray(), [[1, 0, 2, 0, 3, 0]])

    def test_zero_vector_text(self):
        vec = SparseVector({}, 3)
        assert vec.to_list() == SparseVector.from_text(vec.to_text()).to_list()

    def test_from_text(self):
        vec = SparseVector.from_text('{1:1.5,3:2,5:3}/6')
        assert vec.dimensions() == 6
        assert vec.indices() == [0, 2, 4]
        assert vec.values() == [1.5, 2, 3]
        assert vec.to_list() == [1.5, 0, 2, 0, 3, 0]
        assert np.array_equal(vec.to_numpy(), [1.5, 0, 2, 0, 3, 0])

    def test_from_binary(self):
        data = pack('>iii3i3f', 6, 3, 0, 0, 2, 4, 1.5, 2, 3)
        vec = SparseVector.from_binary(data)
        assert vec.dimensions() == 6
        assert vec.indices() == [0, 2, 4]
        assert vec.values() == [1.5, 2, 3]
        assert vec.to_list() == [1.5, 0, 2, 0, 3, 0]
        assert np.array_equal(vec.to_numpy(), [1.5, 0, 2, 0, 3, 0])
        assert vec.to_binary() == data

    def test_to_text(self):
        vec = SparseVector([1, 0, 2, 0, 3, 0])
        assert vec.to_text() == '{1:1.0,3:2.0,5:3.0}/6'

    def test_to_db_none(self):
        assert SparseVector._to_db(None) is None

    def test_to_db_vector(self):
        vec = SparseVector([1, 0, 2, 0, 3, 0])
        assert SparseVector._to_db(vec) == '{1:1.0,3:2.0,5:3.0}/6'

    def test_to_db_list(self):
        result = SparseVector._to_db([1, 0, 2, 0, 3, 0])
        assert result == '{1:1.0,3:2.0,5:3.0}/6'

    def test_to_db_with_dim(self):
        result = SparseVector._to_db([1, 0, 2, 0, 3, 0], 6)
        assert result == '{1:1.0,3:2.0,5:3.0}/6'

    def test_to_db_wrong_dim(self):
        with pytest.raises(ValueError, match='expected 6 dimensions, not 5'):
            SparseVector._to_db([1, 0, 2, 0, 3], 6)

    def test_to_db_binary_none(self):
        assert SparseVector._to_db_binary(None) is None

    def test_to_db_binary_vector(self):
        vec = SparseVector([1, 0, 2, 0, 3, 0])
        result = SparseVector._to_db_binary(vec)
        assert isinstance(result, bytes)

    def test_to_db_binary_list(self):
        result = SparseVector._to_db_binary([1, 0, 2, 0, 3, 0])
        assert isinstance(result, bytes)

    def test_from_db_none(self):
        assert SparseVector._from_db(None) is None

    def test_from_db_sparsevector(self):
        vec = SparseVector([1, 0, 2, 0, 3, 0])
        assert SparseVector._from_db(vec) is vec

    def test_from_db_text(self):
        result = SparseVector._from_db('{1:1.5,3:2,5:3}/6')
        assert isinstance(result, SparseVector)
        assert result.to_list() == [1.5, 0, 2, 0, 3, 0]

    def test_from_db_binary_none(self):
        assert SparseVector._from_db_binary(None) is None

    def test_from_db_binary_sparsevector(self):
        vec = SparseVector([1, 0, 2, 0, 3, 0])
        assert SparseVector._from_db_binary(vec) is vec

    def test_from_db_binary_bytes(self):
        data = pack('>iii3i3f', 6, 3, 0, 0, 2, 4, 1.5, 2, 3)
        result = SparseVector._from_db_binary(data)
        assert isinstance(result, SparseVector)
        assert result.to_list() == [1.5, 0, 2, 0, 3, 0]

    def test_empty_sparse_vector(self):
        vec = SparseVector({}, 5)
        assert vec.dimensions() == 5
        assert vec.indices() == []
        assert vec.values() == []
        assert vec.to_list() == [0, 0, 0, 0, 0]

    def test_single_nonzero(self):
        vec = SparseVector({3: 42}, 10)
        assert vec.dimensions() == 10
        assert vec.indices() == [3]
        assert vec.values() == [42]

    def test_negative_values_sparse(self):
        vec = SparseVector([-1, 0, -2, 0, -3])
        assert vec.values() == [-1, -2, -3]
        assert vec.to_list() == [-1, 0, -2, 0, -3]

    def test_roundtrip_text_sparse(self):
        original = SparseVector([1.5, 0, 2.5, 0, 3.5, 0])
        text = original.to_text()
        restored = SparseVector.from_text(text)
        assert restored == original

    def test_roundtrip_binary_sparse(self):
        original = SparseVector([1.5, 0, 2.5, 0, 3.5, 0])
        binary = original.to_binary()
        restored = SparseVector.from_binary(binary)
        assert restored == original
