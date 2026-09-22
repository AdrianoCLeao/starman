//! A small worker pool for reimport jobs, with cancellation by generation:
//! every submission for a path bumps that path's generation, so a job that
//! is superseded while queued (or in flight) is skipped or its result is
//! dropped instead of being applied on top of newer content.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::import::ImportOutcome;

#[derive(Clone, Debug)]
pub(crate) struct Job {
    pub path: PathBuf,
    pub relative_path: String,
    pub generation: u64,
}

#[derive(Debug)]
pub(crate) struct JobResult {
    pub path: PathBuf,
    pub generation: u64,
    pub outcome: Result<ImportOutcome, String>,
}

type WorkFn = dyn Fn(&Job) -> Result<ImportOutcome, String> + Send + Sync;

pub(crate) struct JobPool {
    job_tx: Sender<Job>,
    result_rx: Receiver<JobResult>,
    generations: Arc<Mutex<HashMap<PathBuf, u64>>>,
    pending: Arc<AtomicUsize>,
}

impl JobPool {
    pub fn new(worker_threads: usize, work: Arc<WorkFn>) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (result_tx, result_rx) = mpsc::channel::<JobResult>();
        let job_rx = Arc::new(Mutex::new(job_rx));
        let generations: Arc<Mutex<HashMap<PathBuf, u64>>> = Arc::default();
        let pending = Arc::new(AtomicUsize::new(0));

        for index in 0..worker_threads.max(1) {
            let job_rx = Arc::clone(&job_rx);
            let result_tx = result_tx.clone();
            let generations = Arc::clone(&generations);
            let pending = Arc::clone(&pending);
            let work = Arc::clone(&work);

            let spawned = thread::Builder::new()
                .name(format!("asset-reimport-{index}"))
                .spawn(move || loop {
                    let job = {
                        let Ok(receiver) = job_rx.lock() else { return };
                        match receiver.recv() {
                            Ok(job) => job,
                            Err(_) => return,
                        }
                    };

                    if !is_current(&generations, &job.path, job.generation) {
                        pending.fetch_sub(1, Ordering::SeqCst);
                        continue;
                    }

                    let outcome = work(&job);
                    let _ = result_tx.send(JobResult {
                        path: job.path,
                        generation: job.generation,
                        outcome,
                    });
                    pending.fetch_sub(1, Ordering::SeqCst);
                });

            if let Err(error) = spawned {
                log::warn!(target: "engine::assets", "failed to spawn reimport worker: {error}");
            }
        }

        Self {
            job_tx,
            result_rx,
            generations,
            pending,
        }
    }

    /// Queues a reimport of `path`, superseding any earlier job for it.
    pub fn submit(&self, path: PathBuf, relative_path: String) {
        let generation = {
            let Ok(mut generations) = self.generations.lock() else {
                return;
            };
            let entry = generations.entry(path.clone()).or_insert(0);
            *entry += 1;
            *entry
        };

        self.pending.fetch_add(1, Ordering::SeqCst);
        let job = Job {
            path,
            relative_path,
            generation,
        };
        if self.job_tx.send(job).is_err() {
            self.pending.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// Jobs submitted whose result has not been produced (or skipped) yet.
    #[cfg(test)]
    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::SeqCst)
    }

    /// Drains finished jobs without blocking, dropping any result that was
    /// superseded by a newer submission for the same path.
    pub fn poll_results(&self) -> Vec<JobResult> {
        let mut results = Vec::new();
        while let Ok(result) = self.result_rx.try_recv() {
            if is_current(&self.generations, &result.path, result.generation) {
                results.push(result);
            }
        }
        results
    }
}

fn is_current(generations: &Mutex<HashMap<PathBuf, u64>>, path: &PathBuf, generation: u64) -> bool {
    generations
        .lock()
        .map(|generations| generations.get(path).copied() == Some(generation))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn outcome_for(job: &Job) -> Result<ImportOutcome, String> {
        Ok(ImportOutcome {
            path: job.path.clone(),
            relative_path: job.relative_path.clone(),
            content_hash: format!("gen-{}", job.generation),
        })
    }

    fn wait_until_idle(pool: &JobPool) -> Vec<JobResult> {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut collected = Vec::new();
        loop {
            collected.extend(pool.poll_results());
            if pool.pending() == 0 {
                collected.extend(pool.poll_results());
                return collected;
            }
            assert!(Instant::now() < deadline, "job pool did not become idle");
            thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn a_single_job_is_delivered() {
        let pool = JobPool::new(2, Arc::new(|job: &Job| outcome_for(job)));
        pool.submit(PathBuf::from("a.png"), "a.png".to_owned());

        let results = wait_until_idle(&pool);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].generation, 1);
    }

    #[test]
    fn a_superseded_in_flight_job_is_dropped() {
        // One worker, blocked inside the first job until released, so the
        // second submission for the same path lands while the first is in
        // flight.
        let (gate_tx, gate_rx) = mpsc::channel::<()>();
        let gate_rx = Mutex::new(gate_rx);
        let pool = JobPool::new(
            1,
            Arc::new(move |job: &Job| {
                let _ = gate_rx.lock().unwrap().recv();
                outcome_for(job)
            }),
        );

        pool.submit(PathBuf::from("a.png"), "a.png".to_owned());
        pool.submit(PathBuf::from("a.png"), "a.png".to_owned());
        gate_tx.send(()).unwrap();
        gate_tx.send(()).unwrap();

        let results = wait_until_idle(&pool);
        assert_eq!(results.len(), 1, "only the newest generation survives");
        assert_eq!(results[0].generation, 2);
    }

    #[test]
    fn jobs_for_different_paths_do_not_supersede_each_other() {
        let pool = JobPool::new(2, Arc::new(|job: &Job| outcome_for(job)));
        pool.submit(PathBuf::from("a.png"), "a.png".to_owned());
        pool.submit(PathBuf::from("b.png"), "b.png".to_owned());

        let results = wait_until_idle(&pool);
        assert_eq!(results.len(), 2);
    }
}
