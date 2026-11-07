import protocol
import collect
from schema import validate_row


if __name__ == '__main__':
    collect.validate_row = validate_row
    collect.main()
