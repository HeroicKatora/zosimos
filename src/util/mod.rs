/// An object-safe version of `Extend`.
pub(crate) trait ExtendOne<Item> {
    fn extend_one(&mut self, item: Item);
    fn extend_iter(&mut self, iter: &mut dyn Iterator<Item=Item>) {
        for item in iter {
            self.extend_one(item);
        }
    }
}

impl<Item, T: Extend<Item> + ?Sized> ExtendOne<Item> for T {
    fn extend_one(&mut self, item: Item) {
        // TODO: use `extend_one` here as well.
        Extend::extend(self, core::iter::once(item))
    }

    fn extend_iter(&mut self, iter: &mut dyn Iterator<Item=Item>) {
        Extend::extend(self, iter)
    }
}
