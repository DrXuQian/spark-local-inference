// SPDX-License-Identifier: Apache-2.0

//! Replay external actlizeLA kernels after preparation and Python have exited.

mod allocation_watch;

use flare_helios::{ReplayWorker, runtime_package::NativeResources};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path, time::Duration};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
struct Case {
    name: String,
    graph: String,
    uploads: BTreeMap<String, String>,
    expected: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct Expected {
    kind: String,
    addresses: BTreeMap<String, usize>,
    buffers: BTreeMap<String, Buffer>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Buffer {
    allocation: String,
    offset: usize,
    nbytes: usize,
}

fn replay(
    root: &Path,
    expected: &Expected,
    owner: &NativeResources,
    worker: &ReplayWorker,
) -> Result<()> {
    for case in &expected.cases {
        let graph = owner
            .graph_names()
            .iter()
            .position(|name| name == &case.graph)
            .ok_or("missing graph")?;
        let mut uploads = BTreeMap::new();
        for (name, file) in &case.uploads {
            let view = &expected.buffers[name];
            if !uploads.contains_key(&view.allocation) {
                uploads.insert(view.allocation.clone(), owner.download(&view.allocation)?);
            }
            let bytes = uploads.get_mut(&view.allocation).unwrap();
            let input = std::fs::read(root.join(file))?;
            assert_eq!(input.len(), view.nbytes);
            bytes[view.offset..view.offset + view.nbytes].copy_from_slice(&input);
        }
        for (name, bytes) in uploads {
            owner.upload(&name, &bytes)?;
        }
        let ticket = worker.submit(graph, 1)?;
        let result = worker
            .poll(Duration::from_secs(30))?
            .ok_or("native replay timed out")?;
        assert_eq!(ticket, result.ticket);
        if let Some(error) = result.error {
            return Err(error.into());
        }
        let mut compared = 0;
        let mut downloaded = BTreeMap::new();
        for (name, file) in &case.expected {
            let view = &expected.buffers[name];
            if !downloaded.contains_key(&view.allocation) {
                downloaded.insert(view.allocation.clone(), owner.download(&view.allocation)?);
            }
            let bytes = &downloaded[&view.allocation][view.offset..view.offset + view.nbytes];
            let wanted = std::fs::read(root.join(file))?;
            assert!(bytes == wanted, "{}: byte mismatch in {name}", case.name);
            compared += bytes.len();
        }
        println!("{}: {compared} bytes match after native replay", case.name);
    }
    Ok(())
}

fn main() -> Result<()> {
    let directory = std::env::args_os()
        .nth(1)
        .ok_or("expected fixture directory")?;
    let root = Path::new(&directory);
    let expected: Expected = serde_json::from_slice(&std::fs::read(root.join("expected.json"))?)?;
    let count = match expected.kind.as_str() {
        "actlize-gdn-export-v1" => 15,
        "actlize-qsa-export-v1" => 17,
        _ => return Err("unknown actlize qualification fixture".into()),
    };
    assert_eq!(expected.cases.len(), count);
    let allocations = allocation_watch::AllocationWatch::new()?;
    let owner = unsafe {
        NativeResources::load(
            flare_helios::runtime_package::test_support::authorize(root.join("package")),
            0,
        )?
    };
    for (name, address) in &expected.addresses {
        assert_eq!(
            owner.allocation(name)?.address,
            *address,
            "{name} did not restore its prepared virtual address"
        );
    }
    let worker = unsafe { ReplayWorker::new(owner.graph_bundle(), 1)? };
    let baseline = allocations.count();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
        replay(root, &expected, &owner, &worker)?;
        // Repeat changed inputs and all prefix lengths on the same allocations.
        replay(root, &expected, &owner, &worker)?;
        assert_eq!(
            allocations.count(),
            baseline,
            "execution allocated CUDA storage"
        );
        Ok(())
    }));
    if let Ok(Err(error)) = &result {
        eprintln!("Native replay failed: {error}");
    }
    let shutdown = worker.close();
    if !shutdown.drained {
        std::mem::forget(owner);
        return Err(format!("native worker did not drain: {:?}", shutdown.error).into());
    }
    if let Some(error) = shutdown.error {
        return Err(error.into());
    }
    match result {
        Ok(result) => result?,
        Err(panic) => std::panic::resume_unwind(panic),
    }
    let maps = std::fs::read_to_string("/proc/self/maps")?;
    assert!(
        !maps.contains("libpython")
            && !maps.contains("libtorch")
            && !maps.contains("libactlize_la_serving")
    );
    println!(
        "{} native {} replays passed; no Python, Torch or operator host library",
        2 * count,
        expected.kind
    );
    Ok(())
}
