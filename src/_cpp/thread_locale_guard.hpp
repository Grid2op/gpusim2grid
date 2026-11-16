// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// thread_locale_guard.hpp -- give the calling thread its locale back
// =============================================================================
//
// cuDSS JIT-compiles kernels through NVRTC on a process' first cudssExecute,
// and nvrtcCompileProgram switches the calling thread to the "C" locale
// (uselocale) without switching it back. The thread then reads ASCII as its
// codeset: in Python, locale.getpreferredencoding() becomes ASCII and every
// later open() without an explicit encoding fails on a non-ASCII file.
//
// ThreadLocaleGuard saves the thread's locale (uselocale(0): the per-thread
// one, or LC_GLOBAL_LOCALE when the thread uses the global locale) and puts it
// back when it goes out of scope. Wrap every call that may reach NVRTC.
// Pinned in tests/python/test_thread_locale.py.
// =============================================================================

#ifndef GPUSIM2GRID_THREAD_LOCALE_GUARD_HPP
#define GPUSIM2GRID_THREAD_LOCALE_GUARD_HPP

#include <locale.h>

class ThreadLocaleGuard {
public:
    ThreadLocaleGuard() : saved_(uselocale(static_cast<locale_t>(0))) {}
    ~ThreadLocaleGuard() {
        if (saved_ != static_cast<locale_t>(0)) uselocale(saved_);
    }
    ThreadLocaleGuard(const ThreadLocaleGuard&) = delete;
    ThreadLocaleGuard& operator=(const ThreadLocaleGuard&) = delete;

private:
    locale_t saved_;
};

#endif  // GPUSIM2GRID_THREAD_LOCALE_GUARD_HPP
