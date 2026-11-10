import protocol
import collect
import pair_schema
from schema import validate_row,per_pair_enabled


if __name__=='__main__':
    pair_schema.enabled=per_pair_enabled
    collect.validate_row=validate_row
    collect.main()
