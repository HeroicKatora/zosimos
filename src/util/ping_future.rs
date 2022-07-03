use std::future::Future;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

pub struct Ping<T> {
    complete: Arc<State>,
    with: Box<dyn Fn() -> T>,
}

pub struct PingWaker {
    complete: Arc<State>,
}

#[derive(Default)]
struct State {
    waiter: Condvar,
    wake: Mutex<Shared>,
}

#[derive(Default)]
struct Shared {
    done: u8,
    waker: Option<Waker>,
}

impl<T> Ping<T> {
    pub fn new(with: Box<dyn Fn() -> T>) -> (Self, PingWaker) {
        let complete = Arc::default();
        let ping = Ping {
            complete: Arc::clone(&complete),
            with: with.into(),
        };
        let waker = PingWaker { complete };
        (ping, waker)
    }

    #[allow(dead_code)]
    pub fn wait_while(&self, with: impl FnMut() -> bool) -> Option<Result<(), T>> {
        match self.complete.wait_while(with) {
            None => None,
            Some(true) => Some(Ok(())),
            Some(false) => Some(Err((self.with)())),
        }
    }
}

impl PingWaker {
    pub fn complete(self, success: bool) {
        self.complete.complete(success)
    }
}

impl State {
    pub(crate) fn complete(&self, success: bool) {
        let mut lock = self.wake.lock().unwrap();

        if let Some(w) = &lock.waker {
            w.wake_by_ref();
        }

        lock.done = !success as u8 + 1;
        self.waiter.notify_all();
    }

    pub(crate) fn wait_while(&self, mut with: impl FnMut() -> bool) -> Option<bool> {
        self.wait_for_inner(&mut with)
    }

    fn wait_for_inner(&self, with: &mut dyn FnMut() -> bool) -> Option<bool> {
        let lock = self.wake.lock().unwrap();
        let lock = self
            .waiter
            .wait_while(lock, |state: &mut Shared| state.done == 0 || with())
            .unwrap();
        match lock.done {
            0 => None,
            1 => Some(true),
            _ => Some(false),
        }
    }
}

impl<T> Future for Ping<T> {
    type Output = Result<(), T>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut lock = self.complete.wake.lock().unwrap();

        match lock.done {
            0 => {}
            1 => return Poll::Ready(Ok(())),
            _ => return Poll::Ready(Err((self.with)())),
        }

        lock.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}
