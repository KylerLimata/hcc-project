use std::path::{Path, PathBuf};
use godot::classes::ProjectSettings;
use godot::prelude::GString;
use pyo3::{Bound, PyAny, PyResult, Python};
use pyo3::prelude::PyAnyMethods;

pub fn get_venv_path() -> std::path::PathBuf {
    let ps = ProjectSettings::singleton();
    let project_root = ps.globalize_path(&GString::from("res://")).to_string();

    PathBuf::from(project_root).join("hcc-python")
}

pub fn add_site_packages(py: Python, venv_path: &Path) -> PyResult<()> {
    let sys = py.import("sys")?;

    #[cfg(target_os = "windows")]
    let site_packages = venv_path.join("Lib").join("site-packages");
    #[cfg(not(target_os = "windows"))]
    let site_packages = venv_path.join("lib").join("python3.10").join("site-packages"); // adjust as needed

    // Get sys.path as &PyAny
    let path: Bound<PyAny> = sys.getattr("path")?;
    // Call append on it
    path.call_method1("append", (site_packages.to_str().unwrap(),))?;

    Ok(())
}