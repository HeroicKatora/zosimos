use crate::pool::Pool;

pub enum LaunchError {
}

pub struct Execution {
}

pub enum RetireError {
}

impl Execution {
    pub fn retire(self, pool: &mut Pool) -> Result<(), RetireError> {
        todo!()
    }
}
