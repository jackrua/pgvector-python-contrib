import numpy as np
from pgvector import Bit
import pytest


class TestBit:
    def test_list(self):
        assert Bit([True, False, True]).to_list() == [True, False, True]

    def test_list_none(self):
        with pytest.warns(UserWarning, match='expected elements to be boolean'):
            assert Bit([True, None, True]).to_text() == '101'

    def test_list_int(self):
        with pytest.warns(UserWarning, match='expected elements to be boolean'):
            assert Bit([254, 7, 0]).to_text() == '110'

    def test_tuple(self):
        assert Bit((True, False, True)).to_list() == [True, False, True]

    def test_str(self):
        assert Bit('101').to_list() == [True, False, True]

    def test_bytes(self):
        assert Bit(b'\xff\x00\xf0').to_text() == '111111110000000011110000'
        assert Bit(b'\xfe\x07\x00').to_text() == '111111100000011100000000'

    def test_ndarray(self):
        arr = np.array([True, False, True])
        assert Bit(arr).to_list() == [True, False, True]
        assert np.array_equal(Bit(arr).to_numpy(), arr)

    def test_ndarray_unpackbits(self):
        arr = np.unpackbits(np.array([254, 7, 0], dtype=np.uint8))
        assert Bit(arr).to_text() == '111111100000011100000000'

    def test_ndarray_uint8(self):
        arr = np.array([254, 7, 0], dtype=np.uint8)
        with pytest.warns(UserWarning, match='expected elements to be boolean'):
            assert Bit(arr).to_text() == '110'

    def test_ndarray_uint16(self):
        arr = np.array([254, 7, 0], dtype=np.uint16)
        with pytest.warns(UserWarning, match='expected elements to be boolean'):
            assert Bit(arr).to_text() == '110'

    def test_ndim_two(self):
        with pytest.raises(ValueError) as error:
            Bit([[True, False], [True, False]])
        assert str(error.value) == 'expected ndim to be 1'

    def test_ndim_zero(self):
        with pytest.raises(ValueError) as error:
            Bit(True)
        assert str(error.value) == 'expected ndim to be 1'

    def test_repr(self):
        assert repr(Bit([True, False, True])) == 'Bit(101)'
        assert str(Bit([True, False, True])) == 'Bit(101)'

    def test_equality(self):
        assert Bit([True, False, True]) == Bit([True, False, True])
        assert Bit([True, False, True]) != Bit([True, False, False])

    def test_equality_with_different_type(self):
        assert Bit([True, False, True]) != [True, False, True]
        assert Bit([True, False, True]) != "not a bit"
        assert Bit([True, False, True]) != None

    def test_length(self):
        assert len(Bit([True, False, True]).to_list()) == 3
        assert len(Bit('10101010').to_list()) == 8

    def test_to_text(self):
        assert Bit([True, False, True]).to_text() == '101'
        assert Bit('10101010').to_text() == '10101010'

    def test_from_text(self):
        bit = Bit.from_text('101')
        assert bit.to_list() == [True, False, True]

    def test_to_binary(self):
        bit = Bit([True, False, True])
        binary = bit.to_binary()
        assert isinstance(binary, bytes)
        assert len(binary) > 0

    def test_from_binary(self):
        bit = Bit([True, False, True, False, True, False, True, False])
        binary = bit.to_binary()
        restored = Bit.from_binary(binary)
        assert restored == bit

    def test_from_binary_error(self):
        with pytest.raises(ValueError, match='expected bytes'):
            Bit.from_binary('not bytes')

    def test_to_db(self):
        bit = Bit([True, False, True])
        assert Bit._to_db(bit) == '101'

    def test_to_db_error(self):
        with pytest.raises(ValueError, match='expected bit'):
            Bit._to_db([True, False, True])

    def test_to_db_binary(self):
        bit = Bit([True, False, True])
        result = Bit._to_db_binary(bit)
        assert isinstance(result, bytes)

    def test_to_db_binary_error(self):
        with pytest.raises(ValueError, match='expected bit'):
            Bit._to_db_binary([True, False, True])

    def test_empty_bit(self):
        bit = Bit([])
        assert bit.to_list() == []
        assert bit.to_text() == ''

    def test_single_bit(self):
        bit = Bit([True])
        assert bit.to_list() == [True]
        assert bit.to_text() == '1'

    def test_bytes_constructor(self):
        bit = Bit(b'\x01')
        assert len(bit.to_list()) == 8
        assert bit.to_text() == '00000001'

    def test_roundtrip_text(self):
        original = Bit('10110011')
        text = original.to_text()
        restored = Bit.from_text(text)
        assert restored == original

    def test_roundtrip_binary(self):
        original = Bit([True, False, True, True, False, False, True, True])
        binary = original.to_binary()
        restored = Bit.from_binary(binary)
        assert restored == original
