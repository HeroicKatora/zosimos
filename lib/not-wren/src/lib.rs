//! A causally consistent, multi-client, garbage collecting key-value ''store''.
//!
//! The map will share its keys and values with all readers and writers through immutable data
//! structures. Additionally, all operations will maintain a consistent view of the value IDs
//! present in the map by returning differential updates of those sets. (If an operation does not
//! return such sets, and does not require a callback handling such sets, then it does not modify
//! the value set).
//!
//! Local write clients need to update their state periodically for cleanup (not lock nor hazard-free).
//! 
//! Not distributed but in theory very much ready for distributed extensions. We also only have one
//! data partition.

/// The map itself won't store values, only reference count identifiers.
pub struct ValId(u64);

pub struct Map<K: Hash + Eq> {

}


impl<K> Map<K> {
    pub fn new(reader: usize, writer: usize)
        -> (Self, Vec<Reader>, Vec<Writer>)
    {
        todo!()
    }
}

impl<K: Hash + Eq> Map {

}
