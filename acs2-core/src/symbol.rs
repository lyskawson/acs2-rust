#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Symbol {
    Wildcard,
    Token(u8),
}

impl Symbol {
    pub fn is_wildcard(self) -> bool {
        todo!()
    }

    pub fn token(self) -> Option<u8> {
        todo!()
    }
}
