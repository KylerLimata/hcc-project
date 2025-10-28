use std::f64::consts::{FRAC_PI_4, PI};
use godot::classes::{IVehicleBody3D, VehicleBody3D};
use godot::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyTuple};
use pyo3::{pyclass, pymethods, Bound, Py, PyAny, PyResult, Python};
use std::ffi::CString;
use std::fs;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use tokio::runtime::{Handle, Runtime};
use tokio::task::JoinHandle;

const DEGREES_30_RADIANS: f64 = 30.0 * PI / 180.0;

struct HCCPythonRunnerExtension;

#[gdextension]
unsafe impl ExtensionLibrary for HCCPythonRunnerExtension {}

#[derive(GodotClass)]
#[class(base=Node)]
struct PythonScriptRunner {
    msg_receiver: Option<Receiver<Message>>,
    episode_handle: Option<EpisodeHandle>,
    python_task_handle: Option<JoinHandle<()>>,
    tokio_runtime: Runtime,
    base: Base<Node>
}

#[godot_api]
impl INode for PythonScriptRunner {
    fn init(base: Base<Node>) -> Self {
        Self {
            msg_receiver: None,
            episode_handle: None,
            python_task_handle: None,
            tokio_runtime: Runtime::new().expect("Failed to create Tokio runtime"),
            base
        }
    }

    fn physics_process(&mut self, _delta: f64) {
        if let Some(receiver) = self.msg_receiver.as_mut() {
            let messages: Vec<_> = receiver.try_iter().collect(); // borrow ends here
            for message in messages {
                match message {
                    Message::Print(s) => godot_print!("{}", s),
                    Message::LoadEnvironment(ref name) => {
                        godot_print!("Loading environment...");
                        self.signals().load_environment().emit(&GString::from(name));
                    }
                    Message::RunEpisode(agent, max_steps, episode_result) => {
                        let wrapped = PythonAgent::new(agent);
                        self.signals().run_episode().emit(&wrapped, max_steps);
                        self.episode_handle = Some(episode_result);
                    }
                }
            }
        }

        let mut is_task_finished = false;

        if let Some(handle) = self.python_task_handle.as_mut() {
            is_task_finished = handle.is_finished()
        }
        if is_task_finished {
            self.python_task_handle = None;
        }
    }
}

#[godot_api]
impl PythonScriptRunner {
    #[func]
    fn run_script(&mut self, name: GString) {
        godot_print!("Running `{}.py`", name);

        let file_contents = match fs::read_to_string(format!("{}.py", name)) {
            Ok(contents) => contents,
            Err(err) => {
                godot_error!("Error loading python script: {:?}", err);
                return;
            }
        };
        let (tx, rx): (Sender<Message>, Receiver<Message>) = mpsc::channel();
        let handle: Handle = self.tokio_runtime.handle().clone();
        self.msg_receiver = Some(rx);

        Python::initialize();

        let join_handle = handle.spawn(async move {
            let simulation_runner = SimulationRunner {
                message_sender: tx
            };

            match Python::attach(|py| {
                let locals = PyDict::new(py);
                locals.set_item("sim", simulation_runner)?;

                py.run(&*CString::new(file_contents).unwrap(), None, Some(&locals))
            }) {
                Ok(_) => {
                    godot_print!("Script ran successfully!");
                }
                Err(err) => {
                    godot_error!("Error running python script: {:?}", err);
                }
            }
        });
        self.python_task_handle = Some(join_handle);
    }

    #[func]
    fn complete_episode(&mut self, checkpoint_times: Array<i64>, terminated: bool) {
        match &mut self.episode_handle {
            None => godot_error!("Attempted to return episode result when no episode is running."),
            Some(handle) => {
                handle.set_result((checkpoint_times.iter_shared().collect(), terminated))
            }
        }
    }

    #[signal]
    fn load_environment(path: GString);

    #[signal]
    fn evaluate_agent(agent: Gd<PythonAgent>);

    #[signal]
    fn run_episode(agent: Gd<PythonAgent>, max_steps: i64);
}

#[derive(GodotClass)]
#[class(base=VehicleBody3D)]
struct AgentVehicleBody {
    #[var]
    agent: Option<Gd<PythonAgent>>,
    base: Base<VehicleBody3D>,
    distances: Vec<f64>
}

#[godot_api]
impl IVehicleBody3D for AgentVehicleBody {
    fn init(base: Base<Self::Base>) -> Self {
        Self {
            agent: None,
            base,
            distances: vec![]
        }
    }

    fn physics_process(&mut self, _delta: f64) {
        let global_transform = self.base_mut().get_global_transform();
        let velocity = self.base().get_linear_velocity();
        let forward = global_transform.basis.col_c();
        let speed = velocity.dot(forward) as f64;
        let steering_angle = self.base_mut().get_steering() as f64;

        if let Some(agent) = self.agent.as_mut() {
            let outputs: Vec<f32> = Python::attach(|py| {
                let distances = if self.distances.is_empty() { vec![5.0, 5.0, 5.0] } else { self.distances.clone() };
                let state = vec![speed, steering_angle];
                let args = (distances, state);
                let pyclass = agent.bind_mut();

                pyclass.agent.call_method1(py, "eval", args).unwrap().extract(py).unwrap()
            });

            let engine_force_multiplier = outputs.get(0).unwrap();
            let steering_direction = outputs.get(1).unwrap().clamp(-1.0, 1.0) as f64;
            let steering_angle = steering_angle + steering_direction*PI/180.0;

            self.base_mut().set_engine_force(*engine_force_multiplier*25.0);
            self.base_mut().set_steering(steering_angle.clamp(-DEGREES_30_RADIANS, DEGREES_30_RADIANS) as f32);
        }
    }
}

#[godot_api]
impl AgentVehicleBody {
    #[func]
    fn attach_agent(&mut self, agent: Gd<PythonAgent>) {
        self.agent = Some(agent);
    }

    #[func]
    fn update_raycast_distances(&mut self, distances    : Array<f64>) {
        self.distances = distances.iter_shared().collect();
    }
}

#[derive(GodotClass)]
#[class(no_init, base=Resource)]
struct PythonAgent {
    agent: Py<PyAny>,
    base: Base<Resource>
}

#[godot_api]
impl IResource for PythonAgent {
}

impl PythonAgent {
    fn new(agent: Py<PyAny>) -> Gd<Self> {
        Gd::from_init_fn(|base| {
            Self {
                agent,
                base
            }
        })
    }
}

#[pyclass]
struct SimulationRunner {
    message_sender: Sender<Message>
}

#[pymethods]
impl SimulationRunner {
    fn print(&mut self, string: String) -> PyResult<()> {
        match self.message_sender.send(Message::Print(string)) {
            Ok(_) => Ok(()),
            Err(_) => Err(pyo3::exceptions::PyOSError::new_err("Failed to print output."))
        }
    }

    fn load_environment(&mut self, name: String) -> PyResult<()> {
        match self.message_sender.send(Message::LoadEnvironment(name)) {
            Ok(_) => Ok(()),
            Err(_) => Err(pyo3::exceptions::PyOSError::new_err("Failed to load environment."))
        }
    }

    fn run_episode(&mut self, instance: &Bound<'_, PyAny>, max_steps: i64) -> PyResult<EpisodeHandle> {
        let episode_result = EpisodeHandle::new();
        let episode_result_clone = episode_result.clone();

        match self.message_sender.send(Message::RunEpisode(instance.clone().unbind(), max_steps, episode_result_clone)) {
            Ok(_) => (),
            Err(err) => return Err(pyo3::exceptions::PyOSError::new_err(err.to_string()))
        };

        Ok(episode_result)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct EpisodeHandle {
    inner: Arc<(Mutex<Option<(Vec<i64>, bool)>>, Condvar)>,
}

#[pymethods]
impl EpisodeHandle {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new((Mutex::new(None), Condvar::new()))
        }
    }

    fn is_done(&self) -> bool {
        let (lock, _) = &*self.inner;
        lock.lock().unwrap().is_some()
    }

    fn set_result(&self, value: (Vec<i64>, bool)) {
        let (lock, cvar) = &*self.inner;
        let mut guard = lock.lock().unwrap();
        *guard = Some(value);
        cvar.notify_all();
    }

    fn get_result(&self) -> PyResult<(Vec<i64>, bool)> {
        let (lock, cvar) = &*self.inner;
        let mut guard = lock.lock().unwrap();
        while guard.is_none() {
            guard = cvar.wait(guard).unwrap();  // block until result is ready
        }
        Ok(guard.as_ref().unwrap().clone())
    }
}

enum Message {
    Print(String),
    LoadEnvironment(String),
    RunEpisode(Py<PyAny>, i64, EpisodeHandle),
}

