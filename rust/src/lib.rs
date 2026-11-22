// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: 2026 Rémy O

use pyo3::prelude::*;

mod algebra;

#[pymodule]
mod pyroclastic_rust {
    use pyo3::prelude::*;

    #[pymodule]
    mod algebra {
        use pyo3::prelude::*;

        #[pyfunction]
        fn h_approx(d: num_bigint::BigInt, b: u32) -> f64 {
            crate::algebra::h_approx(&d, b)
        }

        #[pyfunction]
        fn smallprimes(b: u32) -> Vec<u32> {
            crate::algebra::smallprimes(b)
        }
    }
}
