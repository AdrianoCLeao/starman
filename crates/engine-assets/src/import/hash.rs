//! Content hashing, used to tell a real content change apart from a mere
//! mtime touch (e.g. a checkout, or an editor re-saving identical bytes).

pub(crate) fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_bytes_hash_the_same() {
        assert_eq!(hash_bytes(b"hello"), hash_bytes(b"hello"));
    }

    #[test]
    fn different_bytes_hash_differently() {
        assert_ne!(hash_bytes(b"hello"), hash_bytes(b"world"));
    }
}
