mod ping_future;

/// An object-safe version of `Extend`.
pub(crate) trait ExtendOne<Item> {
    fn extend_one(&mut self, item: Item);
}

impl<Item, T: Extend<Item> + ?Sized> ExtendOne<Item> for T {
    fn extend_one(&mut self, item: Item) {
        // TODO: use `extend_one` here as well.
        Extend::extend(self, core::iter::once(item))
    }
}

pub use ping_future::Ping;
