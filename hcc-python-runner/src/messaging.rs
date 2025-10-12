use std::sync::{mpsc};
use std::sync::mpsc::{SendError, RecvError};
use crate::messaging::SendRecvError::{RecvErr, SendErr};

pub(crate) struct SenderReceiver<S,R> {
    pub sender: mpsc::Sender<S>,
    pub receiver: mpsc::Receiver<R>
}

impl<S, R> SenderReceiver<S, R> {
    fn channels() -> (Self, SenderReceiver<R, S>) {
        let (sender1, receiver1) = mpsc::channel();
        let (sender2, receiver2) = mpsc::channel();
        let self1 = Self { sender: sender1, receiver: receiver2 };
        let self2 = SenderReceiver { sender: sender2, receiver: receiver1 };

        (self1, self2)
    }

    fn send_and_listen(&mut self, msg: S) -> Result<R, SendRecvError<S>> {
        match self.sender.send(msg) {
            Ok(_) => {
                match self.receiver.recv() {
                    Ok(response) => Ok(response),
                    Err(err) => Err(RecvErr(err))
                }
            }
            Err(err) => Err(SendErr(err))
        }
    }

    fn receive_and_respond(&mut self, func: fn(R) -> S) -> Result<(), SendRecvError<S>> {
        match self.receiver.recv() {
            Ok(msg) => {
                let response = func(msg);
                match self.sender.send(response) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(SendErr(err))
                }
            },
            Err(err) => Err(RecvErr(err))
        }
    }

    fn try_recv(&mut self) -> Result<R, mpsc::TryRecvError> {
        self.receiver.try_recv()
    }

    fn send(&mut self, msg: S) -> Result<(), SendError<S>> {
        self.sender.send(msg)
    }
}

pub(crate) enum SendRecvError<S> {
    SendErr(SendError<S>),
    RecvErr(RecvError),
}