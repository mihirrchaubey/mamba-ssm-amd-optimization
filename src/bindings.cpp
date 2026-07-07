#include <torch/extension.h>
#include "selective_scan.h"

// Define the binding module name. This must exactly match the name 
// given in your pybind11_add_module directive in CMakeLists.txt.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",               // The function name you'll use in Python (e.g., mamba_kernel.forward)
        &selective_scan_fwd_hip, // Pointer to the C++/HIP function we declared in the header
        "Mamba Selective Scan Forward Pass (AMD HIP Optimized)", // Docstring
        py::arg("u"),
        py::arg("delta"),
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("D"),
        py::arg("z") = py::none(),          // Default optional value
        py::arg("delta_bias") = py::none(), // Default optional value
        py::arg("is_variable_B") = true
    );
}
