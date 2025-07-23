#include <carma>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "apart.h"

namespace py = pybind11;


PYBIND11_MODULE(_core, m)
{
    m.def("apart",
          &apart,
          "Apply APART algorithm for change point detection (VAR model)",
          py::arg("signal"),
          py::arg("pen") = 1.0,
          py::arg("n_states") = 1);
}