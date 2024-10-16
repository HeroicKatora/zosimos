use std::time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
pub struct TimeAccountant {
    host_start: Instant,
    time_spent: Duration,
}

#[cfg(target_arch = "wasm32")]
pub struct TimeAccountant {
    _inner: (),
}

#[cfg(not(target_arch = "wasm32"))]
impl TimeAccountant {
    pub fn from_now() -> Self {
        TimeAccountant {
            host_start: Instant::now(),
            time_spent: Default::default(),
        }
    }

    pub fn checkpoint(&mut self) {
        let new_now = std::time::Instant::now();
        let took_time = new_now.saturating_duration_since(self.host_start);
        // FIXME: also add 'skip', after asynchronous waits a different measure for the time in
        // these must be taken and that interval should be discarded from time spent.
        self.time_spent += took_time;
        self.host_start = new_now;
    }

    pub fn spent(&self) -> Duration {
        self.time_spent
    }
}

#[cfg(target_arch = "wasm32")]
impl TimeAccountant {
    pub fn from_now() -> Self {
        TimeAccountant { _inner: () }
    }

    pub fn checkpoint(&mut self) {}

    pub fn spent(&self) -> Duration {
        Duration::default()
    }
}
