from sqlalchemy.dialects.postgresql.asyncpg import PGDialect_asyncpg
from sqlalchemy.dialects.postgresql.base import ischema_names
from sqlalchemy.types import UserDefinedType, Float
from .. import Bit 

import asyncpg

class BIT(UserDefinedType):
    cache_ok = True

    def __init__(self, length=None):
        super(UserDefinedType, self).__init__()
        self.length = length

    def get_col_spec(self, **kw):
        if self.length is None:
            return 'BIT'
        return 'BIT(%d)' % self.length

    def bind_processor(self, dialect):
        def process(value):
            if value is None: 
                return None
            val = Bit._to_db(value)
            if isinstance(dialect, PGDialect_asyncpg): 
                return asyncpg.BitString(val)
            return val
        return process

    # def literal_processor(): 
        ... # TODO

    # def result_processor():
        ... # TODO You need this one to pass the tests

    class comparator_factory(UserDefinedType.Comparator):
        def hamming_distance(self, other):
            return self.op('<~>', return_type=Float)(other)

        def jaccard_distance(self, other):
            return self.op('<%>', return_type=Float)(other)


# for reflection
ischema_names['bit'] = BIT
