use std::sync::Arc;

use tokio::sync::{mpsc, oneshot};

/// Task for the inference worker.
pub(crate) struct InferenceTask<I, O> {
    /// The request to process.
    pub req: I,
    /// The response sender.
    pub resp_tx: oneshot::Sender<O>,
}

/// State for the inference task.
#[derive(Clone)]
pub(crate) struct InferenceState<I, O> {
    pub tx: mpsc::Sender<InferenceTask<I, O>>,
}

/// Function signature for processing inference tasks.
pub(crate) type ProcessFn<P, I, O> = dyn Fn(&P, I) -> O + Send + Sync;

/// Distributes inference tasks to worker tasks that process them using the provided function. The
/// function is expected to be thread-safe and is cloned for each worker.
///
/// # Arguments
///
/// * `rx` - Receiver for incoming inference tasks.
/// * `pipeline` - The inference pipeline that is going to be used to process the tasks.
/// * `num_workers` - The number of worker tasks to spawn.
/// * `process_fn` - The function that processes the inference task.
pub(crate) async fn task_distributor<P, I, O, F>(
    mut rx: mpsc::Receiver<InferenceTask<I, O>>,
    pipeline: Arc<P>,
    num_workers: usize,
    process_fn: Arc<ProcessFn<P, I, O>>,
) where
    P: Send + Sync + 'static,
    I: Send + 'static,
    O: Send + 'static,
    F: Send + ?Sized + 'static,
{
    tracing::info!("Starting task distributor with {} workers", num_workers);
    let mut workers = Vec::new();
    let (worker_tx, mut worker_rx) = mpsc::channel(num_workers);

    // Spawn worker tasks
    for _ in 0..num_workers {
        let worker_pipeline = Arc::clone(&pipeline);
        let worker_tx = worker_tx.clone();
        let worker_process_fn = Arc::clone(&process_fn) as Arc<ProcessFn<P, I, O>>;
        let handle = tokio::spawn(worker_loop::<P, I, O, F>(
            worker_pipeline,
            worker_tx,
            worker_process_fn,
        ));
        workers.push(handle);
    }
    drop(worker_tx);

    let mut tasks = Vec::new();
    let mut available_workers: Vec<mpsc::Sender<InferenceTask<I, O>>> = Vec::new();

    loop {
        tokio::select! {
            task = rx.recv() => {
                match task {
                    Some(task) => {
                        if let Some(worker) = available_workers.pop() {
                            worker.send(task).await.expect("Failed to send task to worker");
                        } else {
                            tasks.push(task);
                        }
                    }
                    None => {
                        // Channel closed, no more tasks will be coming
                        break;
                    }
                }
            }
            worker = worker_rx.recv() => {
                match worker {
                    Some(worker) => {
                        if let Some(task) = tasks.pop() {
                            worker.send(task).await.expect("Failed to send task to worker");
                        } else {
                            available_workers.push(worker);
                        }
                    }
                    None => {
                        // All workers have exited
                        break;
                    }
                }
            }
        }
    }

    // Wait for all workers to complete
    for worker in workers {
        worker.await.expect("Worker task panicked");
    }
}

/// Worker task that processes inference tasks using the provided function.
///
/// # Arguments
///
/// * `id` - Worker task identifier.
/// * `pipeline` - The inference pipeline that is going to be used to process the tasks.
/// * `worker_tx` - Sender to communicate with the task distributor.
/// * `process_fn` - The function that processes the inference task.
async fn worker_loop<P, I, O, F: ?Sized>(
    pipeline: Arc<P>,
    worker_tx: mpsc::Sender<mpsc::Sender<InferenceTask<I, O>>>,
    process_fn: Arc<ProcessFn<P, I, O>>,
) {
    let (task_tx, mut task_rx) = mpsc::channel(1);

    loop {
        if worker_tx.send(task_tx.clone()).await.is_err() {
            break;
        }

        if let Some(task) = task_rx.recv().await {
            let result = process_fn(&pipeline, task.req);
            if task.resp_tx.send(result).is_err() {
                tracing::error!("Failed to send response from worker");
            }
        } else {
            break;
        }
    }
}
