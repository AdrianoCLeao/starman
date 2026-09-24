//! `AssetServer` integration for typed assets: resolving [`AssetRef`]
//! requests through the asset database, decoding on worker threads, and
//! re-running loaders on hot reload.

use std::any::{Any, TypeId};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use engine_core::{HardeningConfig, SourceAssetId, SubAssetId};

use crate::typed::{AssetRef, Assets, LoadContext, LoaderEntry, ResolvedSource};
use crate::{AssetDatabase, AssetId, AssetPath};

/// Outcome of one [`crate::AssetServer::update`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AssetUpdateReport {
    /// `(type name, label)` of every typed asset whose payload (re)loaded.
    pub loaded: Vec<(&'static str, String)>,
    /// `(type name, label, error)` of every failed typed load.
    pub failed: Vec<(&'static str, String, String)>,
}

impl AssetUpdateReport {
    pub fn is_empty(&self) -> bool {
        self.loaded.is_empty() && self.failed.is_empty()
    }

    fn merge(&mut self, other: AssetUpdateReport) {
        self.loaded.extend(other.loaded);
        self.failed.extend(other.failed);
    }
}

struct Job {
    asset_type: TypeId,
    id: AssetId,
    source: ResolvedSource,
    loader: LoaderEntry,
    hardening: HardeningConfig,
}

struct Done {
    asset_type: TypeId,
    id: AssetId,
    label: String,
    result: std::result::Result<Box<dyn Any + Send>, String>,
}

fn run_job(job: Job) -> Done {
    let label = match &job.source.sub_key {
        Some(key) => format!("{}#{key}", job.source.relative_path),
        None => job.source.relative_path.clone(),
    };
    let result = std::fs::read(&job.source.disk_path)
        .map_err(|error| {
            format!(
                "failed to read '{}': {error}",
                job.source.disk_path.display()
            )
        })
        .and_then(|bytes| {
            let mut ctx = LoadContext::new(
                &job.source.disk_path,
                &job.source.relative_path,
                job.source.sub_key.as_deref(),
                &job.hardening,
            );
            (job.loader.load)(&bytes, &mut ctx).map_err(|error| error.to_string())
        });
    Done {
        asset_type: job.asset_type,
        id: job.id,
        label,
        result,
    }
}

/// A small fixed pool decoding assets off the main thread.
pub(crate) struct LoadWorkers {
    jobs: Option<Sender<Box<Job>>>,
    done_rx: Receiver<Done>,
    done_tx: Sender<Done>,
    threads: Vec<JoinHandle<()>>,
    outstanding: usize,
}

impl LoadWorkers {
    pub fn new() -> Self {
        let (done_tx, done_rx) = mpsc::channel();
        Self {
            jobs: None,
            done_rx,
            done_tx,
            threads: Vec::new(),
            outstanding: 0,
        }
    }

    fn ensure_started(&mut self) {
        if self.jobs.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel::<Box<Job>>();
        let rx = Arc::new(Mutex::new(rx));
        let workers = std::thread::available_parallelism()
            .map(|count| count.get().clamp(1, 4))
            .unwrap_or(2);
        for index in 0..workers {
            let rx = Arc::clone(&rx);
            let done = self.done_tx.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("starman-asset-load-{index}"))
                .spawn(move || loop {
                    let job = {
                        let Ok(guard) = rx.lock() else { return };
                        guard.recv()
                    };
                    let Ok(job) = job else { return };
                    if done.send(run_job(*job)).is_err() {
                        return;
                    }
                });
            match spawned {
                Ok(handle) => self.threads.push(handle),
                Err(error) => log::warn!(
                    target: "engine::assets",
                    "failed to spawn asset load worker: {error}"
                ),
            }
        }
        self.jobs = Some(tx);
    }

    fn submit(&mut self, job: Job) {
        self.ensure_started();
        let job = if self.threads.is_empty() {
            Some(job)
        } else {
            match self.jobs.as_ref().map(|tx| tx.send(Box::new(job))) {
                Some(Ok(())) => None,
                Some(Err(mpsc::SendError(job))) => Some(*job),
                None => None,
            }
        };
        if let Some(job) = job {
            // No worker available: decode inline, deliver through the same
            // completion channel.
            let _ = self.done_tx.send(run_job(job));
        }
        self.outstanding += 1;
    }

    fn drain(&mut self, block: bool) -> Vec<Done> {
        let mut out = Vec::new();
        if block {
            while self.outstanding > 0 {
                match self.done_rx.recv() {
                    Ok(done) => {
                        self.outstanding -= 1;
                        out.push(done);
                    }
                    Err(_) => break,
                }
            }
        }
        while let Ok(done) = self.done_rx.try_recv() {
            self.outstanding = self.outstanding.saturating_sub(1);
            out.push(done);
        }
        out
    }
}

impl Drop for LoadWorkers {
    fn drop(&mut self) {
        self.jobs = None;
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Resolves `request` to a concrete file (and sub-asset key).
pub(crate) fn resolve_request(
    root: &AssetPath,
    database: Option<&mut AssetDatabase>,
    request: &AssetRef,
) -> std::result::Result<ResolvedSource, String> {
    let id_text = request.id.trim();
    let (mut relative_path, mut sub_key): (Option<String>, Option<String>) = (None, None);
    let (mut source_id, mut sub_id) = (None, None);

    if !id_text.is_empty() {
        let uuid =
            uuid_parse(id_text).ok_or_else(|| format!("'{id_text}' is not a valid asset id"))?;
        if let Some(database) = database.as_deref() {
            let as_source = SourceAssetId::from_uuid(uuid);
            if let Some(path) = database.resolve_relative_path(as_source) {
                relative_path = Some(path.to_owned());
                source_id = Some(as_source);
            } else {
                let as_sub = SubAssetId::from_uuid(uuid);
                if let Some((parent, record)) = database.resolve_sub_asset(as_sub) {
                    if let Some(path) = database.resolve_relative_path(parent) {
                        relative_path = Some(path.to_owned());
                        sub_key = Some(record.key.clone());
                        source_id = Some(parent);
                        sub_id = Some(as_sub);
                    }
                }
            }
        }
        if relative_path.is_none() && request.path.trim().is_empty() {
            return Err(format!(
                "asset id {id_text} is unknown to the asset database (was the asset deleted?)"
            ));
        }
    }

    if relative_path.is_none() {
        let (file, key) = request.split_path();
        if file.is_empty() {
            return Err("asset reference has neither an id nor a path".to_owned());
        }
        relative_path = Some(file.to_owned());
        sub_key = key.map(str::to_owned);
        if let Some(database) = database {
            if let Ok(id) = database.ensure_imported(file) {
                source_id = Some(id);
                if let Some(key) = &sub_key {
                    sub_id = database
                        .sub_assets_of(id)
                        .iter()
                        .find(|record| &record.key == key)
                        .map(|record| record.id);
                }
            }
        }
    }

    let relative_path = relative_path.expect("resolved above");
    let disk_path = crate::pathing::resolve_disk_path(root, &relative_path)
        .map_err(|error| error.to_string())?;
    if !disk_path.is_file() {
        return Err(format!("asset file '{relative_path}' does not exist"));
    }
    Ok(ResolvedSource {
        relative_path,
        disk_path,
        sub_key,
        source_id,
        sub_id,
    })
}

fn uuid_parse(text: &str) -> Option<uuid::Uuid> {
    uuid::Uuid::parse_str(text).ok()
}

/// Resolves pending requests and dispatches loads. With `block`, decodes
/// on the calling thread and returns only when nothing is in flight.
pub(crate) fn process(
    assets: &Assets,
    workers: &mut LoadWorkers,
    root: &AssetPath,
    mut database: Option<&mut AssetDatabase>,
    hardening: &HardeningConfig,
    block: bool,
) -> AssetUpdateReport {
    let mut report = AssetUpdateReport::default();
    loop {
        let pending = assets.pending_requests();
        for (asset_type, id, request) in &pending {
            let resolved = resolve_request(root, database.as_deref_mut(), request);
            if let Err(error) = &resolved {
                report.failed.push((
                    assets
                        .loader_for_type(*asset_type)
                        .map(|entry| entry.type_name)
                        .unwrap_or("asset"),
                    request.to_string(),
                    error.clone(),
                ));
                log::warn!(target: "engine::assets", "cannot load {request}: {error}");
            }
            assets.resolve_request(*asset_type, *id, resolved);
            dispatch(
                assets,
                workers,
                *asset_type,
                *id,
                hardening,
                block,
                &mut report,
            );
        }
        report.merge(collect(assets, workers, block));
        if !block || !assets.has_in_flight() || pending.is_empty() {
            break;
        }
    }
    report
}

fn dispatch(
    assets: &Assets,
    workers: &mut LoadWorkers,
    asset_type: TypeId,
    id: AssetId,
    hardening: &HardeningConfig,
    inline: bool,
    report: &mut AssetUpdateReport,
) {
    let Some(loader) = assets.loader_for_type(asset_type) else {
        let error = "no loader registered for this asset type".to_owned();
        assets.complete_load(asset_type, id, Err(error.clone()));
        report
            .failed
            .push(("asset", format!("#{}", id.value()), error));
        return;
    };
    let Some(source) = assets.begin_load(asset_type, id) else {
        return;
    };
    let job = Job {
        asset_type,
        id,
        source,
        loader,
        hardening: *hardening,
    };
    if inline {
        let done = run_job(job);
        finish(assets, done, report);
    } else {
        workers.submit(job);
    }
}

fn collect(assets: &Assets, workers: &mut LoadWorkers, block: bool) -> AssetUpdateReport {
    let mut report = AssetUpdateReport::default();
    for done in workers.drain(block) {
        finish(assets, done, &mut report);
    }
    report
}

fn finish(assets: &Assets, done: Done, report: &mut AssetUpdateReport) {
    let error = done.result.as_ref().err().cloned();
    let type_name = assets
        .loader_for_type(done.asset_type)
        .map(|entry| entry.type_name)
        .unwrap_or("asset");
    match assets.complete_load(done.asset_type, done.id, done.result) {
        Some(type_name) => report.loaded.push((type_name, done.label)),
        None => {
            let error = error.unwrap_or_else(|| "load failed".to_owned());
            log::warn!(
                target: "engine::assets",
                "failed to load {type_name} '{}': {error}",
                done.label
            );
            report.failed.push((type_name, done.label, error));
        }
    }
}

/// Re-runs the loader of every typed asset read from `disk_path`.
pub(crate) fn reload_disk_path(
    assets: &Assets,
    workers: &mut LoadWorkers,
    hardening: &HardeningConfig,
    disk_path: &Path,
    block: bool,
) -> usize {
    let slots = assets.slots_for_disk_path(disk_path);
    let mut report = AssetUpdateReport::default();
    for (asset_type, id) in &slots {
        dispatch(
            assets,
            workers,
            *asset_type,
            *id,
            hardening,
            block,
            &mut report,
        );
    }
    slots.len()
}

pub(crate) fn collect_finished(assets: &Assets, workers: &mut LoadWorkers) -> AssetUpdateReport {
    collect(assets, workers, false)
}

/// Normalizes a watcher path for comparison with resolved sources.
pub(crate) fn normalized(path: &Path) -> PathBuf {
    crate::pathing::normalize_disk_path(path)
}
