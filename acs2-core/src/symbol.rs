#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Symbol {
    Wildcard,
    Token(u8),
}

impl Symbol {
    pub fn is_wildcard(self) -> bool {
        matches!(self, Symbol::Wildcard)
    }

    pub fn token(self) -> Option<u8> {
        match self {
            Symbol::Wildcard => None,
            Symbol::Token(value) => Some(value),
        }
    }
}
