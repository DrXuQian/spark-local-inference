// SPDX-License-Identifier: Apache-2.0

//! Native replay floor for the exported partial Qwen GGUF fixture.
//! Upload preparation, admission, prefill and state readback stay outside timing.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use flare_helios::{
    execution::{FrameCompletion, GraphExecutor, OwnedUpload, Upload},
    runtime_package::{KvCacheLimit, LoadOptions, MemoryLimit, NativeModel, test_support},
    startup::StartupProgress,
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[derive(Deserialize)]
struct Bytes {
    buffer: String,
    offset: usize,
    data: String,
}
#[derive(Deserialize)]
struct Step {
    graph: String,
    uploads: Vec<Bytes>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    init: Step,
    prefill: Step,
    round: Step,
    width: usize,
    budget: usize,
    expected_tokens: Vec<i64>,
}
#[derive(Deserialize)]
struct Corpus {
    model_id: String,
    draft_tokens: usize,
    cases: Vec<Case>,
}
struct Bound {
    graph: usize,
    uploads: Vec<OwnedUpload>,
}
fn bind(executor: &GraphExecutor, step: &Step) -> Result<Bound> {
    let graph = executor
        .manifest()
        .graphs
        .iter()
        .position(|g| g.name == step.graph)
        .ok_or("missing graph")?;
    let uploads = step
        .uploads
        .iter()
        .map(|u| {
            Ok(OwnedUpload {
                buffer: u.buffer.clone(),
                offset: u.offset,
                bytes: STANDARD.decode(&u.data)?,
            })
        })
        .collect::<Result<_>>()?;
    Ok(Bound { graph, uploads })
}
fn borrowed(step: &Bound) -> Vec<Upload<'_>> {
    step.uploads
        .iter()
        .map(|u| Upload {
            buffer: &u.buffer,
            offset: u.offset,
            bytes: &u.bytes,
        })
        .collect()
}
fn one(executor: &mut GraphExecutor, step: &Bound) -> Result<FrameCompletion> {
    assert_eq!(executor.outstanding(), 0);
    let fence = executor
        .submit(step.graph, &borrowed(step), false)?
        .ok_or("unexpected backpressure")?;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(c) = executor.poll()? {
            assert_eq!(c.fence, fence);
            return Ok(c);
        }
        if Instant::now() > deadline {
            return Err("replay timeout".into());
        }
        std::hint::spin_loop();
    }
}
fn tokens(c: &FrameCompletion) -> Result<Vec<i64>> {
    let r = &c.records[0];
    if r.error_code != 0 || r.status == 3 {
        return Err(format!("model error: {:?}", r).into());
    }
    let v = r.vector.as_ref().ok_or("missing completion vector")?;
    Ok(v.tokens[..v.emitted_count as usize].to_vec())
}
// Each replay has a fresh request generation and submission fences. Check those
// independently; every other completion field must match exactly.
fn record(c: &FrameCompletion) -> Vec<i64> {
    let mut r = c.records[0].fields();
    r[0] = 0;
    r[1] = 0;
    r
}
fn snapshot(executor: &GraphExecutor) -> Result<Vec<(String, Vec<u8>)>> {
    // This corpus assigns target/draft inputs to the first 48 KV pages.
    // Copy that complete range and the committed live state row after drain.
    let cuda = unsafe { libloading::Library::new("libcuda.so.1")? };
    let sync: libloading::Symbol<unsafe extern "C" fn() -> i32> =
        unsafe { cuda.get(b"cuCtxSynchronize\0")? };
    let copy: libloading::Symbol<unsafe extern "C" fn(*mut u8, u64, usize) -> i32> =
        unsafe { cuda.get(b"cuMemcpyDtoH_v2\0")? };
    assert_eq!(unsafe { sync() }, 0);
    let mut values = Vec::new();
    for b in &executor.manifest().buffers {
        let (offset, nbytes) = if b.name == "state_pool" {
            (b.nbytes / b.shape[0], b.nbytes / b.shape[0])
        } else if b.name.starts_with("cache.") {
            (0, b.nbytes / b.shape[0] * 48)
        } else {
            continue;
        };
        assert!(offset + nbytes <= b.nbytes);
        let mut bytes = vec![0u8; nbytes];
        let code = unsafe { copy(bytes.as_mut_ptr(), (b.address + offset) as u64, nbytes) };
        if code != 0 {
            return Err(format!("download {}: CUDA {code}", b.name).into());
        }
        values.push((b.name.clone(), bytes));
    }
    assert_eq!(values.len(), 40); // state plus 13 QSA layers' three declared KV buffers
    Ok(values)
}
fn generation(step: &mut Bound, name: &str, value: i64) -> Result<()> {
    let upload = step
        .uploads
        .iter_mut()
        .find(|u| u.buffer == name)
        .ok_or("missing generation input")?;
    assert_eq!(upload.bytes.len(), 8);
    upload.bytes = value.to_le_bytes().to_vec();
    Ok(())
}
fn run(executor: &mut GraphExecutor, corpus: &Corpus, output: &PathBuf) -> Result<()> {
    let mut report = Vec::new();
    for (index, case) in corpus.cases.iter().enumerate() {
        let epoch = 2 * index as i64 + 1;
        let mut init = bind(executor, &case.init)?;
        let mut prefill = bind(executor, &case.prefill)?;
        generation(&mut init, "admission_generations", epoch)?;
        generation(&mut prefill, "row_generations", epoch)?;
        one(executor, &init)?;
        let first = one(executor, &prefill)?;
        assert_eq!(first.records[0].generation, epoch);
        let initial = tokens(&first)?;
        assert_eq!(initial.len(), 1);
        let mut expected = initial.clone();
        let mut rounds = Vec::new();
        let mut reference = Vec::new();
        let mut reference_logprobs = Vec::new();
        while expected.len() < case.budget {
            let mut step = bind(executor, &case.round)?;
            let limit = step
                .uploads
                .iter_mut()
                .find(|u| u.buffer == "query_lengths")
                .ok_or("no token limit")?;
            limit.bytes = (case.width.min(case.budget - expected.len()) as i32)
                .to_le_bytes()
                .to_vec();
            let c = one(executor, &step)?;
            assert_eq!(c.records[0].generation, epoch);
            let ids = tokens(&c)?;
            if ids.is_empty() {
                return Err("isolated replay made no progress".into());
            }
            expected.extend(ids);
            reference.push(record(&c));
            reference_logprobs.push(c.logprobs);
            rounds.push(step);
            assert!(rounds.len() <= case.budget);
        }
        assert_eq!(
            expected, case.expected_tokens,
            "isolated native replay differs from service"
        );
        assert_eq!(reference.last().unwrap()[5], 2, "request did not finish");
        let before = snapshot(executor)?;
        generation(&mut init, "admission_generations", epoch + 1)?;
        generation(&mut prefill, "row_generations", epoch + 1)?;
        one(executor, &init)?;
        let fresh = one(executor, &prefill)?;
        assert_eq!(fresh.records[0].generation, epoch + 1);
        assert_eq!(tokens(&fresh)?, initial);
        let uploads: Vec<_> = rounds.iter().map(borrowed).collect();
        let mut completed = 0;
        let mut submitted = 0;
        let mut actual = initial;
        let mut actual_logprobs = Vec::with_capacity(rounds.len());
        let mut fences = Vec::with_capacity(rounds.len());
        let mut samples = Vec::with_capacity(rounds.len());
        let mut actual_records = Vec::with_capacity(rounds.len());
        let started = Instant::now();
        let deadline = started + Duration::from_secs(30);
        while completed < rounds.len() {
            while submitted < rounds.len() && executor.outstanding() < 2 {
                let Some(fence) =
                    executor.submit(rounds[submitted].graph, &uploads[submitted], false)?
                else {
                    break;
                };
                fences.push(fence);
                submitted += 1;
            }
            if let Some(c) = executor.poll()? {
                let at = started.elapsed().as_nanos() as u64;
                assert_eq!(c.fence, fences[completed]);
                assert_eq!(c.records[0].generation, epoch + 1);
                actual.extend(tokens(&c)?);
                actual_records.push(record(&c));
                samples.push(json!({"at_ns":at,"gpu_ns":c.timing.gpu_ns[0],
                    "readiness_ns":c.timing.readiness.map(|r|r.wait_ns()),"queued":c.timing.queued,
                    "record":actual_records.last().unwrap()}));
                actual_logprobs.push(c.logprobs);
                completed += 1;
            }
            if Instant::now() > deadline {
                return Err("queued replay timeout".into());
            }
            std::hint::spin_loop();
        }
        let elapsed = started.elapsed().as_nanos() as u64;
        assert_eq!(actual, expected);
        assert_eq!(actual_records, reference);
        assert_eq!(actual_logprobs, reference_logprobs);
        let after = snapshot(executor)?;
        let mut compared = 0;
        for ((an, a), (bn, b)) in before.iter().zip(&after) {
            assert_eq!(an, bn);
            assert!(a == b, "state mismatch: {an}");
            compared += a.len();
        }
        assert_eq!(before.len(), after.len());
        // Retain exact state fingerprints for comparisons across re-exports.
        // Readback and hashing happen after replay has drained, outside timing.
        let fingerprints: std::collections::BTreeMap<_, _> = after
            .iter()
            .map(|(name, bytes)| (name, format!("{:x}", Sha256::digest(bytes))))
            .collect();
        println!(
            "{}: {} rounds, {} ns, {} compared state bytes",
            case.name,
            rounds.len(),
            elapsed,
            compared
        );
        report.push(
            json!({"case":case.name,"rounds":rounds.len(),"elapsed_ns":elapsed,
            "state_bytes_compared":compared,"state_sha256":fingerprints,
            "state_equal":true,"logprobs_equal":true,"tokens":actual,"samples":samples}),
        );
        std::fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    }
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    assert_eq!(args.len(), 3);
    let corpus: Corpus = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let options = LoadOptions {
        embedding_table_storage: Some(flare_helios::engram::Storage::Ram),
        memory_limit: Some(MemoryLimit::Bytes(24 << 30)),
        kv_cache_limit: KvCacheLimit::Memory,
        mtp_draft_tokens: Some(corpus.draft_tokens),
        adaptive_verification: Some(false),
        ..Default::default()
    };
    // This binary accepts only the repository's explicitly known test publisher.
    let owner = unsafe {
        NativeModel::load(
            test_support::authorize(&args[0]),
            &[0],
            &options,
            &StartupProgress::stderr_rows(),
        )?
    };
    assert_eq!(owner.server_config()["model_id"], corpus.model_id);
    assert_eq!(corpus.model_id, "qwen38-gguf-partial-qualification");
    assert_eq!(corpus.draft_tokens, 5);
    let mut manifests = owner.manifests()?;
    assert_eq!(manifests.len(), 1);
    let mut executor = unsafe { GraphExecutor::new(manifests.remove(0))? };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(&mut executor, &corpus, &args[2])
    }));
    let closed = executor.close();
    if !closed.drained {
        std::mem::forget(owner);
        return Err("native replay did not drain".into());
    }
    if let Some(e) = closed.error {
        return Err(e.into());
    }
    match result {
        Ok(result) => result?,
        Err(panic) => std::panic::resume_unwind(panic),
    };
    let maps = std::fs::read_to_string("/proc/self/maps")?;
    assert!(!maps.contains("libpython") && !maps.contains("libtorch"));
    Ok(())
}
