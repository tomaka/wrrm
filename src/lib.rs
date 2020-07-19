// Copyright (c) 2020 Pierre Krieger
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::{boxed::Box, sync::Arc};

use atomicbox_nostd::AtomicOptionBox;
use core::{fmt, ops::{Deref, DerefMut}, sync::atomic::Ordering};

/// "Write-rarely-read-many" wrapper.
///
/// This lock-free container is suitable in situations where you perform a lot of reads to a `T`,
/// but only rarely modify that `T`.
///
/// From a logic point of view, it is more or less the equivalent of an `RwLock`, except that:
///
/// - It works in `no_std` platforms.
/// - Reading the `T` always takes the same time and will never wait for a lock to be released.
/// - Writing the `T` is done in a compare-and-swap way, and updates might have to be performed
/// multiple times.
///
/// # Implementation details
///
/// This container contains more or less the equivalent of an `Atomic<Arc<T>>`. Accessing `T` is
/// quite cheap, as it only consists in cloning the `Arc`. Modifying `T` can be done by performing
/// a deep clone of `T`, then storing a pointer to the updated version in the `Atomic`.
///
/// If `N` threads try to update the `T` at the same time, the entire update of `T` might need to
/// be performed more than `N` times.
/// For example, if two threads try to update `T` at the same time, one of them will win and
/// perform the update. When trying to apply its own update, the other thread will detect that
/// the `T` has been touched in-between and will restart its own update from scratch.
pub struct Wrrm<T> {
    /// If the `AtomicOptionBox` contains `None`, it represents a "lock".
    inner: AtomicOptionBox<Arc<T>>,
}

impl<T> Wrrm<T> {
    /// Creates a new [`Wrrm`].
    pub fn new(value: T) -> Self {
        Wrrm {
            inner: AtomicOptionBox::new(Some(Box::new(Arc::new(value))))
        }
    }

    /// Grants shared access to the content.
    ///
    /// This [`Access`] struct will always point to the same, potentially stale, version. In other
    /// words, if the content of the [`Wrrm`] is updated while an [`Access`] is alive, this
    /// [`Access`] will still point to the old version.
    pub fn access(&self) -> Access<T> {
        let inner = loop {
            if let Some(value) = self.inner.take(Ordering::AcqRel) {
                let _updated = self.inner.swap(Some(value.clone()), Ordering::AcqRel);
                debug_assert!(_updated.is_none());
                break *value;
            }
        };

        Access {
            parent: self,
            inner,
        }
    }

    /// Modifies the value using the given function.
    ///
    /// > **Important**: The function might be called multiple times.
    pub fn modify_with(&self, modification: impl FnMut(&mut T))
    where
        T: Clone,
    {
        self.access().modify_with(modification)
    }
}

impl<T: Default> Default for Wrrm<T> {
    fn default() -> Self {
        Wrrm::new(Default::default())
    }
}

impl<T> From<T> for Wrrm<T> {
    fn from(value: T) -> Self {
        Wrrm::new(value)
    }
}

impl<T> fmt::Debug for Wrrm<T>
where
    T: fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.access(), f)
    }
}

/// Shared access to the content of the [`Wrrm`].
pub struct Access<'a, T> {
    parent: &'a Wrrm<T>,
    inner: Arc<T>,
}

impl<'a, T: Clone> Access<'a, T> {
    /// Modifies the value using the given function.
    ///
    /// > **Important**: The function might be called multiple times.
    pub fn modify_with(self, mut modification: impl FnMut(&mut T)) {
        let mut me = self;

        loop {
            let mut modify = Modify::from(me);
            modification(&mut *modify);
            match Modify::apply(modify) {
                Ok(()) => return,
                Err(acc) => me = acc,
            }
        }
    }
}

impl<'a, T> Deref for Access<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

impl<'a, T> fmt::Debug for Access<'a, T>
where
    T: fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <T as fmt::Debug>::fmt(&*self.inner, f)
    }
}

/// Pending modification to the content of the [`Wrrm`].
pub struct Modify<'a, T> {
    parent: &'a Wrrm<T>,
    /// Value expected to be found in the atomic pointer when writing `new_value`.
    expected: usize,
    new_value: T,
}

impl<'a, T: Clone> From<Access<'a, T>> for Modify<'a, T> {
    fn from(access: Access<'a, T>) -> Self {
        Modify {
            parent: access.parent,
            expected: Arc::as_ptr(&access.inner) as usize,
            new_value: (*access.inner).clone(),
        }
    }
}

impl<'a, T> Modify<'a, T> {
    /// Tries to apply the modifications. Returns an `Err` if the value has been updated
    /// in-between by something else.
    pub fn apply(me: Self) -> Result<(), Access<'a, T>> {
        loop {
            if let Some(in_ptr) = me.parent.inner.take(Ordering::AcqRel) {
                if Arc::as_ptr(&*in_ptr) as usize == me.expected {
                    let new_value = Box::new(Arc::new(me.new_value));
                    let _updated = me.parent.inner.swap(Some(new_value), Ordering::AcqRel);
                    debug_assert!(_updated.is_none());
                    return Ok(())

                } else {
                    let _updated = me.parent.inner.swap(Some(in_ptr.clone()), Ordering::AcqRel);
                    debug_assert!(_updated.is_none());
                    return Err(Access {
                        parent: me.parent,
                        inner: *in_ptr,
                    });
                }
            }
        }
    }
}

impl<'a, T> Deref for Modify<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.new_value
    }
}

impl<'a, T> DerefMut for Modify<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.new_value
    }
}

impl<'a, T> fmt::Debug for Modify<'a, T>
where
    T: fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <T as fmt::Debug>::fmt(&self.new_value, f)
    }
}

#[cfg(test)]
mod tests {
    use super::Wrrm;
    use std::{sync::{Arc, Barrier}, thread, time::Duration};

    #[test]
    fn basic() {
        let val = Wrrm::from(5);

        let first_access = val.access();
        assert_eq!(*first_access, 5);

        val.access().modify_with(|v| *v = 6);

        assert_eq!(*val.access(), 6);
        assert_eq!(*first_access, 5);
    }

    #[test]
    fn threads() {
        let val = Arc::new(Wrrm::from(5));
        let barrier = Arc::new(Barrier::new(256));

        for _ in 0..256 {
            let val = val.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                val.modify_with(|v| *v += 1);
            });
        }

        loop {
            thread::sleep(Duration::from_millis(200));
            if *val.access() == 261 {
                break;
            }
        }

        // Check the value again a bit later, to make sure the number of updates is exactly the
        // one we expect.
        thread::sleep(Duration::from_secs(3));
        assert_eq!(*val.access(), 261);
    }
}
