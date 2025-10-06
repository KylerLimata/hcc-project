use pyo3::Python;

fn main() {
    if std::hint::black_box(false) {
        Python::initialize();
    }
}