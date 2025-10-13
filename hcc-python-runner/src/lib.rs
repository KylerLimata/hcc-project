mod messaging;

use std::ffi::CString;
use std::fs;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use godot::classes::{IVehicleBody3D, VehicleBody3D};
use godot::prelude::*;
use pyo3::{pyclass, pymethods, Bound, Py, PyAny, PyResult, Python};
use pyo3::types::{PyDict, PyDictMethods, PyTuple};

struct HCCPythonRunnerExtension;

#[gdextension]
unsafe impl ExtensionLibrary for HCCPythonRunnerExtension {}

#[derive(GodotClass)]
#[class(base=Node)]
struct PythonScriptRunner {
    print_receiver: Option<Receiver<Message>>,
    agent_return_sender: Option<Sender<Py<PyAny>>>,
    base: Base<Node>
}

#[godot_api]
impl INode for PythonScriptRunner {
    fn init(base: Base<Node>) -> Self {
        Self {
            print_receiver: None,
            agent_return_sender: None,
            base
        }
    }

    fn physics_process(&mut self, delta: f64) {
        if let Some(receiver) = self.print_receiver.as_mut() {
            let messages: Vec<_> = receiver.try_iter().collect(); // borrow ends here
            for message in messages {
                match message {
                    Message::Print(s) => godot_print!("{}", s),
                    Message::EvaluateAgent(agent) => {
                        godot_print!("Agent received");
                        let wrapped = PythonAgent::new(agent);
                        self.signals().evaluate_agent().emit(&wrapped)
                    },
                    Message::LoadEnvironment(ref name) => {
                        godot_print!("Loading environment...");
                        self.signals().load_environment().emit(&GString::from(name));
                    }
                    Message::RunEpisode(agent, max_steps, return_sender) => {
                        let wrapped = PythonAgent::new(agent);
                        self.signals().run_episode().emit(&wrapped, max_steps);
                        self.agent_return_sender = Some(return_sender);
                    }
                }
            }
        }
    }
}

#[godot_api]
impl PythonScriptRunner {
    #[func]
    fn run_script(&mut self, file: GString) {
        godot_print!("loading python script `{}`", file);

        let file_contents = match fs::read_to_string(format!("{}", file)) {
            Ok(contents) => contents,
            Err(err) => {
                godot_error!("Error loading python script: {:?}", err);
                return;
            }
        };
        let (tx, rx): (Sender<Message>, Receiver<Message>) = mpsc::channel();
        let simulation_runner = SimulationRunner {
            message_sender: tx,
        };
        self.print_receiver = Some(rx);

        Python::initialize();

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
    }

    #[func]
    fn return_agent(&mut self, mut python_agent: Gd<PythonAgent>) {
        let bound = python_agent.bind_mut();
        let cloned = Python::attach(|py| bound.agent.clone_ref(py));

        self.agent_return_sender.as_mut().unwrap().send(cloned).unwrap();
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

    fn physics_process(&mut self, delta: f64) {


        if let Some(agent) = self.agent.as_mut() {
            let outputs: Vec<f32> = Python::attach(|py| {

                let distances = if self.distances.is_empty() { vec![5.0, 5.0, 5.0] } else { self.distances.clone() };
                let args = PyTuple::new(py, [distances]).unwrap();
                let pyclass = agent.bind_mut();

                pyclass.agent.call_method1(py, "eval_step", args).unwrap().extract(py).unwrap()
            });

            self.base_mut().set_engine_force(*outputs.get(0).unwrap());
            self.base_mut().set_steering(*outputs.get(1).unwrap());
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
    message_sender: Sender<Message>,
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

    fn evaluate_agent(&mut self, py: Python, instance: &Bound<'_, PyAny>) -> PyResult<()> {
        match self.message_sender.send(Message::EvaluateAgent(instance.clone().unbind())) {
            Ok(_) => Ok(()),
            Err(_) => Err(pyo3::exceptions::PyOSError::new_err("Failed to register agent"))
        }
    }

    fn run_episode(&mut self, py: Python, instance: &Bound<'_, PyAny>, max_steps: i64) -> PyResult<Py<PyAny>> {
        let (tx, rx): (Sender<Py<PyAny>>, Receiver<Py<PyAny>>) = mpsc::channel();
        match self.message_sender.send(Message::RunEpisode(instance.clone().unbind(), max_steps, tx)) {
            Ok(_) => (),
            Err(err) => return Err(pyo3::exceptions::PyOSError::new_err(err.to_string()))
        };

        match rx.recv() {
            Ok(agent) => Ok(agent),
            Err(_) => Err(pyo3::exceptions::PyOSError::new_err("Channel closed!"))
        }
    }
}

enum Message {
    Print(String),
    LoadEnvironment(String),
    EvaluateAgent(Py<PyAny>),
    RunEpisode(Py<PyAny>, i64, Sender<Py<PyAny>>),
}

